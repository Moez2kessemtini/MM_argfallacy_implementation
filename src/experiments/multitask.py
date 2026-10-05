"""Multi-task learning: fallacy task + argumentative sentence detection (MM-USED), with a lambda=0 control.

Usage:
  python -m src.experiments.multitask --task afc --aux asd --seeds 42 2024 666
"""
import argparse
import logging
import zipfile

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torch import nn

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.collators import TextTransformerCollator
from mamkit.data.datasets import InputMode, MMUSEDFallacy
from mamkit.models.text import Transformer
from mamkit.utility.data import download

from src.data.prepare import _alias_numpy2_modules
from src.paths import BASE_DATA_PATH
from src.training.imbalance import inverse_frequency_weights
from src.training.loop import SHARED_TASK_SPLIT, split_shared_task_dialogues
from src.training.runner import load_config, result_dir
from src.utils import enable_determinism, macro_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = {'afc': 6, 'afd': 2}
MMUSED_ARCHIVE_URL = 'https://zenodo.org/api/records/14938592/files-archive'  # same as mamkit's MMUSED loader
MMUSED_DIR = BASE_DATA_PATH / 'MMUSED'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=list(NUM_CLASSES), required=True)
    parser.add_argument('--aux', choices=['asd', 'none'], default='asd')
    parser.add_argument('--aux-weight', type=float, default=0.5, help='lambda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 2024, 666])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--accumulate', type=int, default=3)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--save-model', action='store_true')
    return parser.parse_args()


def run_name(args) -> str:
    suffix = 'none' if args.aux == 'none' else f'{args.aux}{args.aux_weight:g}'
    return f'text_only_roberta_ft_imb-weighted_train_mtl-{suffix}'


# ---------------------------------------------------------------- data

def load_mmused_asd() -> pd.DataFrame:
    """MM-USED sentences with argumentative sentence detection labels (text only)."""
    pkl = MMUSED_DIR / 'dataset.pkl'
    if not pkl.exists():
        MMUSED_DIR.mkdir(parents=True, exist_ok=True)
        archive = MMUSED_DIR / 'data.zip'
        download(url=MMUSED_ARCHIVE_URL, file_path=archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(MMUSED_DIR)
        inner = MMUSED_DIR / 'MMUSED.zip'
        with zipfile.ZipFile(inner) as z:
            z.extractall(MMUSED_DIR)
        archive.unlink()
        inner.unlink()
    try:
        df = pd.read_pickle(pkl)
    except ModuleNotFoundError as e:
        if 'numpy._core' not in str(e):
            raise
        _alias_numpy2_modules()
        df = pd.read_pickle(pkl)
    labels = df['component'].isin(['Premise', 'Claim']).astype(int)
    return pd.DataFrame({'text': df['speech'].astype(str).values, 'label': labels.values,
                         'dialogue_id': df['dialogue_id'].astype(str).values})


def fallacy_frame(task: str) -> pd.DataFrame:
    data = MMUSEDFallacy(task_name=task, input_mode=InputMode.TEXT_ONLY, base_data_path=BASE_DATA_PATH).data
    if task == 'afc':
        data = data[data['fallacy'].notna()]
        return pd.DataFrame({'text': data['snippet'].astype(str).values, 'label': data['fallacy'].astype(int).values,
                             'dialogue_id': data['dialogue_id'].astype(str).values})
    return pd.DataFrame({'text': data['sentence'].astype(str).values, 'label': data['label'].astype(int).values,
                         'dialogue_id': data['dialogue_id'].astype(str).values})


# ---------------------------------------------------------------- model

class MultiTaskText(nn.Module):
    """MAMKit text Transformer with an auxiliary head on the same pooled representation."""

    def __init__(self, config, num_aux_classes: int = 2):
        super().__init__()
        self.backbone = Transformer(model_card=config.model_card, head=config.head,
                                    dropout_rate=config.dropout_rate, is_transformer_trainable=True)
        hidden = self.backbone.model_config.hidden_size
        self.aux_head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, num_aux_classes))

    def pooled(self, input_ids, attention_mask):
        tokens = self.backbone.dropout(self.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
                                       .last_hidden_state)
        return (tokens * attention_mask[:, :, None]).sum(1) / attention_mask.sum(1)[:, None]

    def forward(self, input_ids, attention_mask, head: str = 'main'):
        pooled = self.pooled(input_ids, attention_mask)
        return self.backbone.head(pooled) if head == 'main' else self.aux_head(pooled)


# ---------------------------------------------------------------- training

class Batches:
    """Shuffled mini-batches of (texts, labels), tokenized on the fly; `cycle=True` never ends."""

    def __init__(self, frame, batch_size, collator, device, shuffle=True, cycle=False):
        self.texts, self.labels = list(frame['text']), np.asarray(frame['label'], dtype=int)
        self.batch_size, self.collator, self.device = batch_size, collator, device
        self.shuffle, self.cycle = shuffle, cycle

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __iter__(self):
        while True:
            order = np.random.permutation(len(self.texts)) if self.shuffle else np.arange(len(self.texts))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.collator(inputs=[self.texts[i] for i in idx], context=None)
                yield (enc['inputs'].to(self.device), enc['input_mask'].to(self.device),
                       th.as_tensor(self.labels[idx], device=self.device))
            if not self.cycle:
                return


@th.no_grad()
def predict(model, frame, collator, device, batch_size):
    model.eval()
    logits = [model(ids, mask).float().cpu()
              for ids, mask, _ in Batches(frame, batch_size, collator, device, shuffle=False)]
    return th.cat(logits)


def evaluate(model, frame, collator, device, batch_size, class_weights, task):
    logits = predict(model, frame, collator, device, batch_size)
    labels = th.as_tensor(np.asarray(frame['label'], dtype=int))
    loss = F.cross_entropy(logits, labels, weight=class_weights.cpu()).item()
    pred = logits.argmax(1).numpy()
    if task == 'afd':
        tp = int(((pred == 1) & (labels.numpy() == 1)).sum())
        denom = int((pred == 1).sum() + (labels.numpy() == 1).sum())
        f1 = 2 * tp / denom if denom else 0.0
    else:
        f1 = macro_f1(labels.numpy(), pred)
    return loss, f1, pred


def run_seed(args, seed, config, fallacy, asd, device, out_dir):
    seed_everything(seed)
    _, val_d, test_d = split_shared_task_dialogues(fallacy['dialogue_id'].values)
    logger.info(f'[seed={seed}] validation debates: {val_d}')
    is_test, is_val = fallacy['dialogue_id'].isin(test_d), fallacy['dialogue_id'].isin(val_d)
    train, val, test = fallacy[~is_test & ~is_val], fallacy[is_val], fallacy[is_test]

    num_classes = NUM_CLASSES[args.task]
    main_weights = th.tensor(inverse_frequency_weights(train['label'].to_numpy(), num_classes),
                             dtype=th.float, device=device)
    collator = TextTransformerCollator(model_card=config.model_card, tokenizer_args=config.tokenizer_args)

    seed_everything(seed)
    model = MultiTaskText(config).to(device)
    optimizer = config.optimizer(model.parameters(), **{**config.optimizer_args, 'lr': args.lr})

    aux_iter, aux_weights = None, None
    if args.aux != 'none':
        aux = asd[~asd['dialogue_id'].isin(set(val_d) | set(test_d))]
        aux_weights = th.tensor(inverse_frequency_weights(aux['label'].to_numpy(), 2), dtype=th.float, device=device)
        aux_iter = iter(Batches(aux, config.batch_size, collator, device, cycle=True))
        logger.info(f'[seed={seed}] aux ASD sentences: {len(aux)} (removed {len(asd) - len(aux)} from '
                    f'validation/test debates), positive rate {aux["label"].mean():.3f}')

    best = {'loss': np.inf}
    bad_epochs = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        main_losses, aux_losses = [], []
        optimizer.zero_grad()
        for step, (ids, mask, y) in enumerate(Batches(train, config.batch_size, collator, device), start=1):
            loss = F.cross_entropy(model(ids, mask), y, weight=main_weights)
            main_losses.append(loss.item())
            if aux_iter is not None:
                a_ids, a_mask, a_y = next(aux_iter)
                aux_loss = F.cross_entropy(model(a_ids, a_mask, head='aux'), a_y, weight=aux_weights)
                aux_losses.append(aux_loss.item())
                loss = loss + args.aux_weight * aux_loss
            (loss / args.accumulate).backward()
            if step % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        if step % args.accumulate:  # leftover accumulated gradient of the last incomplete group
            optimizer.step()
            optimizer.zero_grad()

        val_loss, val_f1, _ = evaluate(model, val, collator, device, args.eval_batch_size, main_weights, args.task)
        history.append({'epoch': epoch, 'train_main': float(np.mean(main_losses)),
                        'train_aux': float(np.mean(aux_losses)) if aux_losses else None,
                        'val_loss': val_loss, 'val_f1': val_f1})
        logger.info(f'[seed={seed}] epoch {epoch}: {history[-1]}')
        if val_loss < best['loss']:
            best = {'loss': val_loss, 'f1': val_f1, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best['state'])
    _, test_f1, test_pred = evaluate(model, test, collator, device, args.eval_batch_size, main_weights, args.task)
    np.savez(out_dir / f'test_predictions_seed{seed}_fold0.npz', y_true=test['label'].to_numpy(), y_pred=test_pred)
    if args.save_model:
        th.save(model.backbone.state_dict(), out_dir / f'model_seed{seed}_fold0.pt')
    logger.info(f'[seed={seed}] best epoch {best["epoch"]}  val loss {best["loss"]:.4f}  val F1 {best["f1"]:.4f}  '
                f'test F1 {test_f1:.4f}')
    return {'val_f1': best['f1'], 'test_f1': test_f1, 'best_epoch': best['epoch'], 'history': history}


def main():
    args = parse_args()
    enable_determinism()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    config = load_config(TransformerConfig, ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY,
                                                      task_name=args.task, tags={'anonymous', 'roberta'}))
    fallacy = fallacy_frame(args.task)
    asd = None
    if args.aux != 'none':
        asd = load_mmused_asd()
        shared = sorted(set(asd['dialogue_id']) & set(fallacy['dialogue_id']))
        logger.info(f'MM-USED: {len(asd)} sentences, {asd["dialogue_id"].nunique()} debates, '
                    f'{len(shared)} debate ids shared with MM-USED-Fallacy')
        if not shared:
            raise RuntimeError('no MM-USED debate id matches MM-USED-Fallacy: the validation/test debates could not '
                               'be removed from the auxiliary data -- check the dialogue_id formats')

    out_dir = result_dir(SHARED_TASK_SPLIT, args.task, run_name(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(args, seed, config, fallacy, asd, device, out_dir) for seed in args.seeds]

    test = np.array([r['test_f1'] for r in runs])
    val = np.array([r['val_f1'] for r in runs])
    np.save(out_dir / 'metrics.npy', {
        'validation': {'test_f1': val.tolist(), 'avg_test_f1': (val.mean(), val.std())},
        'test': {'test_f1': test.tolist(), 'avg_test_f1': (test.mean(), test.std())},
        'runs': runs, 'seeds': args.seeds, 'args': {k: str(v) for k, v in vars(args).items()},
    }, allow_pickle=True)
    logger.info(f'{out_dir.name}: test F1 {test.mean():.4f} +- {test.std():.4f}  per seed {np.round(test, 4).tolist()}')


if __name__ == '__main__':
    main()
