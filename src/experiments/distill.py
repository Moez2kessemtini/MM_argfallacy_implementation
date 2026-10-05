"""Text-to-audio distillation (AFC) with raw and supervised controls and label-efficiency read-outs.

Usage:
  python -m src.experiments.distill --seeds 42 2024 666 --teacher-temp 0.02
  python -m src.experiments.distill --seeds 42 2024 666 --mode raw
"""
import argparse
import json
import logging

import numpy as np
import torch as th
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_linear_schedule_with_warmup

from mamkit.configs.audio import TransformerEncoderConfig
from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.collators import TextTransformerCollator
from mamkit.data.datasets import InputMode, MMUSEDFallacy
from mamkit.utility.collators import encode_audio_torch

from src.data.audio_cache import CachedAudioTransformer, _is_cacheable
from src.models.distillation import AudioStudent, QueueDistillationLoss, TextTeacher
from src.paths import BASE_DATA_PATH
from src.training.imbalance import inverse_frequency_weights
from src.training.loop import SHARED_TASK_SPLIT, split_shared_task_dialogues
from src.training.runner import load_config, result_dir
from src.utils import enable_determinism, macro_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK = 'afc'
NUM_CLASSES = 6
READOUTS = ('linear', 'svm')
DEFAULT_TEACHER_TEMP = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--mode', choices=['distill', 'supervised', 'raw'], default='distill')

    distill = parser.add_argument_group('distillation (--mode distill)')
    distill.add_argument('--teacher', choices=['finetuned', 'generic'], default='finetuned')
    distill.add_argument('--teacher-run', default='text_only_roberta_ft_imb-weighted_train',
                         help='AFC run directory holding the fine-tuned teacher weights model_seed<S>_fold0.pt')
    distill.add_argument('--teacher-norm', choices=['standardize', 'none'], default='standardize',
                         help='z-score the teacher pooled features (training-sentence statistics) before its projector')
    distill.add_argument('--distill-data', choices=['sentences', 'afc'], default='sentences',
                         help='sentences = every sentence of the training debates (label-free); '
                              'afc = the training AFC snippets only')
    distill.add_argument('--queue-size', type=int, default=1024)
    distill.add_argument('--teacher-temp', type=float, default=DEFAULT_TEACHER_TEMP)
    distill.add_argument('--student-temp', type=float, default=0.05)
    distill.add_argument('--proj-hidden', type=int, default=2048)
    distill.add_argument('--proj-dim', type=int, default=128)

    train = parser.add_argument_group('optimisation (distill and supervised)')
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--patience', type=int, default=5, help='epochs without validation macro-F1 gain')
    train.add_argument('--batch-size', type=int, default=64)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--weight-decay', type=float, default=0.01)
    train.add_argument('--warmup', type=float, default=0.1, help='fraction of steps with linear LR warm-up')
    train.add_argument('--num-workers', type=int, default=4)

    eff = parser.add_argument_group('label efficiency (trained modes)')
    eff.add_argument('--label-fractions', type=float, nargs='*', default=[],
                     help='also fit the linear read-out on these fractions of the AFC training labels '
                          '(stratified, same subsample for the learned and the raw representations)')
    eff.add_argument('--label-repeats', type=int, default=10, help='random subsamples per fraction')
    return parser.parse_args()


def run_name(args) -> str:
    if args.mode != 'distill':
        return f'audio_only_wavlm_{args.mode}'
    name = f'audio_only_wavlm_comodo-{args.teacher}'
    if args.teacher_norm != 'standardize':
        name += f'_tnorm-{args.teacher_norm}'
    if args.teacher_temp != DEFAULT_TEACHER_TEMP:
        name += f'_tt{args.teacher_temp:g}'
    if args.distill_data != 'sentences':
        name += f'_distill-{args.distill_data}'
    return name


# --------------------------------------------------------------------------- data
class CachedFrames(Dataset):
    """Frozen WavLM frames of audio items, read lazily from the feature cache."""

    def __init__(self, items, extractor: CachedAudioTransformer):
        self.paths = []
        for i, item in enumerate(items):
            if not _is_cacheable(item):
                raise FileNotFoundError(f'missing audio for item {i}: {item}')
            path = extractor._cache_path(item)
            if not path.exists():
                extractor._encode(item)
            self.paths.append(path)
        logger.info(f'{len(self.paths)} audio items ready ({extractor.cache_dir.name})')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return np.load(self.paths[i])


class Pairs(Dataset):
    """(audio frames, target) pairs: target = teacher embedding (distill) or class label (supervised)."""

    def __init__(self, frames: CachedFrames, targets):
        self.frames, self.targets = frames, targets

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i], self.targets[i]


def collate_frames(batch):
    features, mask = encode_audio_torch(batch)
    return {'inputs': features, 'input_mask': mask}


def collate_pairs(batch):
    frames, targets = zip(*batch)
    return collate_frames(frames), th.stack([th.as_tensor(t) for t in targets])


def to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}


# --------------------------------------------------------------------------- representations
@th.no_grad()
def representations(frames: CachedFrames, device, args, student=None) -> np.ndarray:
    """Student pooled representation, or (student=None) masked mean of the raw WavLM frames."""
    if student is not None:
        student.eval()
    loader = DataLoader(frames, batch_size=args.batch_size, shuffle=False, collate_fn=collate_frames,
                        num_workers=args.num_workers)
    out = []
    for batch in loader:
        batch = to_device(batch, device)
        if student is not None:
            out.append(student.represent(batch).cpu().numpy())
        else:
            mask = batch['input_mask'].unsqueeze(-1)
            out.append(((batch['inputs'] * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- teacher
@th.no_grad()
def teacher_embeddings(teacher: TextTeacher, texts, text_config, device, norm: str, batch_size=64):
    collator = TextTransformerCollator(model_card=text_config.model_card, tokenizer_args=text_config.tokenizer_args)
    pooled = []
    for start in range(0, len(texts), batch_size):
        enc = collator(inputs=list(texts[start:start + batch_size]), context=None)
        pooled.append(teacher.pooled(to_device({'inputs': enc['inputs'], 'input_mask': enc['input_mask']}, device)))
    pooled = th.cat(pooled)
    if norm == 'standardize':  # statistics of these (training) sentences only
        teacher.set_standardization(pooled)
    return th.cat([teacher.project(pooled[i:i + 1024]) for i in range(0, len(pooled), 1024)]).cpu()


def similarity_diagnostics(z: th.Tensor, queue_size: int, teacher_temp: float, seed: int) -> dict:
    """Spread of teacher embeddings and entropy of the teacher target over a random queue."""
    rng = np.random.RandomState(seed)
    sample = z[rng.choice(len(z), min(len(z), 2000), replace=False)]
    sims = sample @ sample.T
    off_diag = sims[~th.eye(len(sample), dtype=th.bool)]
    queue = z[rng.choice(len(z), min(len(z), queue_size), replace=False)]
    target = F.softmax(sample[:256] @ queue.T / teacher_temp, dim=1)
    entropy = -(target * target.clamp_min(1e-12).log()).sum(1).mean().item()
    return {'teacher_cosine_mean': off_diag.mean().item(), 'teacher_cosine_std': off_diag.std().item(),
            'teacher_target_entropy': entropy, 'log_queue_size': float(np.log(len(queue)))}


# --------------------------------------------------------------------------- read-outs


def fit_readout(kind: str, x, y, seed: int):
    if kind == 'linear':
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=5000)).fit(x, y)
    counts = np.bincount(y)
    n_splits = max(2, min(3, int(counts[counts > 0].min())))
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced')),
                          {'svc__C': [0.1, 1, 10, 100]}, scoring='f1_macro',
                          cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    return search.fit(x, y).best_estimator_


def evaluate_readouts(reps: dict, labels: dict, seed: int) -> dict:
    results = {}
    for kind in READOUTS:
        readout = fit_readout(kind, reps['train'], labels['train'], seed)
        y_pred = readout.predict(reps['test'])
        results[kind] = {'val_f1': macro_f1(labels['val'], readout.predict(reps['val'])),
                         'test_f1': macro_f1(labels['test'], y_pred), 'y_true': labels['test'], 'y_pred': y_pred}
        logger.info(f'[seed={seed}] read-out {kind}: val {results[kind]["val_f1"]:.4f}  '
                    f'test macro-F1 {results[kind]["test_f1"]:.4f}')
    return results


def stratified_subsample(labels: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Indices keeping `fraction` of every class (at least one example per class)."""
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep.append(rng.choice(idx, max(1, int(round(fraction * len(idx)))), replace=False))
    return np.sort(np.concatenate(keep))


def label_efficiency(learned: dict, raw: dict, labels: dict, fractions, repeats: int, seed: int) -> dict:
    """Linear read-outs on fractions of the training labels, same subsample for learned and raw features."""
    out = {}
    for fraction in fractions:
        rows = []
        for r in range(repeats):
            idx = stratified_subsample(labels['train'], fraction, np.random.default_rng(seed * 1000 + r))
            scores = {}
            for name, reps in (('learned', learned), ('raw', raw)):
                readout = fit_readout('linear', reps['train'][idx], labels['train'][idx], seed)
                scores[name] = macro_f1(labels['test'], readout.predict(reps['test']))
            rows.append({'n_labels': int(len(idx)), **scores})
        learned_f1 = np.array([row['learned'] for row in rows])
        raw_f1 = np.array([row['raw'] for row in rows])
        out[f'{fraction:g}'] = {'n_labels': rows[0]['n_labels'], 'learned': learned_f1.tolist(), 'raw': raw_f1.tolist()}
        logger.info(f'[seed={seed}] labels {fraction:.0%} (n={rows[0]["n_labels"]}): learned {learned_f1.mean():.4f}  '
                    f'raw {raw_f1.mean():.4f}  learned > raw in {np.mean(learned_f1 > raw_f1):.0%} of {repeats} subsamples')
    return out


def train_student(student, train_set: Dataset, first_epoch_idx, step_fn, afc_frames, afc_labels, device, args, seed):
    """Train a student; the epoch is selected by a linear probe on the AFC validation set."""
    params = [p for p in student.parameters() if p.requires_grad]
    steps = len(first_epoch_idx) // args.batch_size + (args.epochs - 1) * (len(train_set) // args.batch_size)
    optimizer = th.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * steps), max(steps, 1))
    generator = th.Generator().manual_seed(seed)

    history, best = [], {'val_f1': -1.0, 'epoch': -1, 'state': None}
    for epoch in range(args.epochs):
        student.train()
        subset = Subset(train_set, first_epoch_idx) if epoch == 0 else train_set
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            collate_fn=collate_pairs, num_workers=args.num_workers, generator=generator)
        stats = []
        for inputs, targets in loader:
            loss, step_stats = step_fn(to_device(inputs, device), targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            stats.append(step_stats)
        epoch_stats = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}

        reps = {k: representations(afc_frames[k], device, args, student) for k in ('train', 'val')}
        val_f1 = macro_f1(afc_labels['val'], fit_readout('linear', reps['train'], afc_labels['train'], seed)
                          .predict(reps['val']))
        history.append({'epoch': epoch, **epoch_stats, 'val_macro_f1': float(val_f1)})
        logger.info(f'[seed={seed}] epoch {epoch}: ' + '  '.join(f'{k} {v:.4f}' for k, v in epoch_stats.items())
                    + f'  val macro-F1 {val_f1:.4f}')
        if val_f1 > best['val_f1']:
            best = {'val_f1': val_f1, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}}
        elif epoch - best['epoch'] >= args.patience:
            logger.info(f'[seed={seed}] early stopping (best epoch {best["epoch"]})')
            break
    student.load_state_dict(best['state'])
    return {'best_epoch': best['epoch'], 'history': history}


def build_distillation(args, seed, train_d, data, text_config, extractor, device):
    afc, sentences = data
    state = None
    if args.teacher == 'finetuned':
        path = result_dir(SHARED_TASK_SPLIT, TASK, args.teacher_run) / f'model_seed{seed}_fold0.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: train the teacher first with --save-model (src.experiments.baseline)')
        state = th.load(path, map_location='cpu')
    teacher = TextTeacher(text_config, args.proj_hidden, args.proj_dim, state).to(device)

    source = sentences if args.distill_data == 'sentences' else afc
    pairs_df = source[source['dialogue_id'].isin(train_d)].reset_index(drop=True)
    z_teacher = teacher_embeddings(teacher, pairs_df['text'].tolist(), text_config, device, args.teacher_norm)
    del teacher
    th.cuda.empty_cache()
    diagnostics = similarity_diagnostics(z_teacher, args.queue_size, args.teacher_temp, seed)
    logger.info(f'[seed={seed}] teacher diagnostics: ' + '  '.join(f'{k} {v:.4f}' for k, v in diagnostics.items()))
    pairs = Pairs(CachedFrames(pairs_df['audio'].tolist(), extractor), z_teacher)
    logger.info(f'[seed={seed}] distillation pairs: {len(pairs)} ({args.distill_data}, training debates)')
    return pairs, z_teacher, diagnostics


def run_seed(seed, args, data, configs, extractor, device):
    afc, _ = data
    text_config, audio_config = configs

    seed_everything(seed)
    train_d, val_d, test_d = split_shared_task_dialogues(afc['dialogue_id'].values)
    logger.info(f'[seed={seed}] validation debates: {val_d}')
    seed_everything(seed)

    afc_frames, afc_labels = {}, {}
    for name, debates in (('train', train_d), ('val', val_d), ('test', test_d)):
        rows = afc[afc['dialogue_id'].isin(debates)]
        afc_frames[name] = CachedFrames(rows['audio'].tolist(), extractor)
        afc_labels[name] = rows['label'].to_numpy()

    info = {'validation_debates': val_d}
    if args.mode == 'raw':
        student = None
    elif args.mode == 'supervised':
        student = AudioStudent(audio_config, args.proj_hidden, args.proj_dim).to(device)
        student.projector.requires_grad_(False)  # unused here: the classifier reads the pooled representation
        head = nn.Linear(audio_config.embedding_dim, NUM_CLASSES).to(device)
        student.add_module('classifier', head)
        weights = th.tensor(inverse_frequency_weights(afc_labels['train'], NUM_CLASSES), dtype=th.float32).to(device)

        def step_fn(inputs, labels):
            loss = F.cross_entropy(head(student.represent(inputs)), labels, weight=weights)
            return loss, {'loss': loss.item()}

        train_set = Pairs(afc_frames['train'], afc_labels['train'])
        info.update(train_student(student, train_set, np.arange(len(train_set)), step_fn, afc_frames, afc_labels,
                                  device, args, seed))
    else:
        pairs, z_teacher, diagnostics = build_distillation(args, seed, train_d, data, text_config, extractor, device)
        student = AudioStudent(audio_config, args.proj_hidden, args.proj_dim).to(device)
        queue_size = min(args.queue_size, len(pairs) - args.batch_size)
        queue_idx = np.random.RandomState(seed).choice(len(pairs), queue_size, replace=False)
        loss_fn = QueueDistillationLoss(z_teacher[queue_idx], args.teacher_temp, args.student_temp).to(device)

        def step_fn(inputs, z_t):
            loss = loss_fn(student(inputs), z_t)
            return loss, loss_fn.last_stats

        # First epoch excludes the queue's own samples, later epochs use every pair.
        first_epoch_idx = np.setdiff1d(np.arange(len(pairs)), queue_idx)
        info.update({'teacher_diagnostics': diagnostics,
                     **train_student(student, pairs, first_epoch_idx, step_fn, afc_frames, afc_labels, device,
                                     args, seed)})

    reps = {k: representations(afc_frames[k], device, args, student) for k in ('train', 'val', 'test')}
    results = evaluate_readouts(reps, afc_labels, seed)
    if student is not None:
        # Complementarity check: read-outs on [raw WavLM mean ; learned representation].
        raw = {k: representations(afc_frames[k], device, args, None) for k in ('train', 'val', 'test')}
        concat = {k: np.concatenate([raw[k], reps[k]], axis=1) for k in reps}
        logger.info(f'[seed={seed}] read-outs on [raw ; learned] ({concat["train"].shape[1]} dims):')
        results.update({f'{kind}-concat': r for kind, r in evaluate_readouts(concat, afc_labels, seed).items()})
        if args.label_fractions:
            info['label_efficiency'] = label_efficiency(reps, raw, afc_labels, args.label_fractions,
                                                        args.label_repeats, seed)
    return results, info


def main():
    args = parse_args()
    enable_determinism()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    text_config = load_config(TransformerConfig, ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY,
                                                           task_name=TASK, tags={'anonymous', 'roberta'}))
    audio_config = load_config(TransformerEncoderConfig, ConfigKey(
        dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name=TASK, tags={'anonymous', 'wavlm'}))
    # Same frozen-feature configuration as the audio baselines -> reuses the feature cache.
    extractor = CachedAudioTransformer(model_card=audio_config.model_card, processor_args=audio_config.processor_args,
                                       model_args=audio_config.model_args, aggregate=audio_config.aggregate,
                                       downsampling_factor=audio_config.downsampling_factor,
                                       sampling_rate=audio_config.sampling_rate)

    afc = MMUSEDFallacy(task_name='afc', input_mode=InputMode.TEXT_AUDIO, base_data_path=BASE_DATA_PATH).data
    afc = afc[afc['fallacy'].notna()].reset_index(drop=True)
    afc = afc.assign(text=afc['snippet'], audio=afc['snippet_paths'], label=afc['fallacy'].astype(int))
    sentences = None
    if args.mode == 'distill' and args.distill_data == 'sentences':
        sentences = MMUSEDFallacy(task_name='afd', input_mode=InputMode.TEXT_AUDIO, base_data_path=BASE_DATA_PATH).data
        sentences = sentences.assign(text=sentences['sentence'], audio=sentences['sentence_path'])

    per_seed, efficiency = {}, {}
    for seed in args.seeds:
        results, info = run_seed(seed, args, (afc, sentences), (text_config, audio_config), extractor, device)
        if 'label_efficiency' in info:
            efficiency[seed] = info['label_efficiency']
        for kind in results:  # 'linear', 'svm' (+ 'linear-concat', 'svm-concat' for trained modes)
            per_seed.setdefault(kind, {'val': [], 'test': []})
            out = result_dir(SHARED_TASK_SPLIT, TASK, f'{run_name(args)}_readout-{kind}')
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f'test_predictions_seed{seed}_fold0.npz',
                     y_true=results[kind]['y_true'], y_pred=results[kind]['y_pred'])
            (out / f'training_history_seed{seed}.json').write_text(json.dumps({**info, 'args': vars(args)}, indent=2))
            per_seed[kind]['val'].append(results[kind]['val_f1'])
            per_seed[kind]['test'].append(results[kind]['test_f1'])

    for kind in per_seed:
        out = result_dir(SHARED_TASK_SPLIT, TASK, f'{run_name(args)}_readout-{kind}')
        test = np.array(per_seed[kind]['test'])
        metrics = {'validation': {'test_f1': per_seed[kind]['val']},
                   'test': {'test_f1': per_seed[kind]['test'], 'avg_test_f1': (test.mean(), test.std())},
                   'seeds': args.seeds, 'args': vars(args)}
        np.save(out / 'metrics.npy', metrics, allow_pickle=True)
        logger.info(f'{out.name}: test macro-F1 {test.mean():.4f} +- {test.std():.4f}  '
                    f'(val {np.mean(per_seed[kind]["val"]):.4f}) over seeds {args.seeds}')

    if efficiency:
        # Paired over (seed, subsample): mean test macro F1 of the linear read-out per label fraction.
        out = result_dir(SHARED_TASK_SPLIT, TASK, f'{run_name(args)}_readout-linear')
        (out / 'label_efficiency.json').write_text(json.dumps(efficiency, indent=2))
        logger.info(f'label efficiency ({run_name(args)} vs raw WavLM, linear read-out, '
                    f'{len(efficiency)} seeds x {args.label_repeats} subsamples):')
        for fraction in efficiency[args.seeds[0]]:
            learned = np.concatenate([efficiency[s][fraction]['learned'] for s in efficiency])
            raw = np.concatenate([efficiency[s][fraction]['raw'] for s in efficiency])
            diff = learned - raw
            logger.info(f'  {float(fraction):>5.0%} labels (n={efficiency[args.seeds[0]][fraction]["n_labels"]}): '
                        f'learned {learned.mean():.4f} +- {learned.std():.4f}  raw {raw.mean():.4f} +- {raw.std():.4f}  '
                        f'diff {diff.mean():+.4f}  learned > raw in {np.mean(diff > 0):.0%}')


if __name__ == '__main__':
    main()
