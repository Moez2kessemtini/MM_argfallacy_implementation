"""Audio-only AFC by cascade: Whisper transcripts classified by a trained text model.

Usage:
  python -m src.experiments.cascade --text-run text_only_roberta_ft_imb-weighted_train --seeds 42 2024 666
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch as th

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.collators import TextTransformerCollator
from mamkit.data.datasets import InputMode, MMUSEDFallacy
from mamkit.models.text import Transformer

from src.data.audio_cache import _clip_id
from src.paths import BASE_DATA_PATH, REPO_ROOT
from src.training.loop import SHARED_TASK_SPLIT, SHARED_TASK_TEST_DIALOGUES
from src.training.runner import load_config, result_dir
from src.utils import macro_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK = 'afc'
DEFAULT_TRANSCRIPTS = REPO_ROOT / 'data_exploration' / 'outputs' / 'e3_alignment_whisper' / 'tables' / \
    'whisper_afc_transcripts.jsonl'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--text-run', default='text_only_roberta_ft_imb-weighted_train',
                        help='AFC text run directory holding model_seed<S>_fold0.pt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--transcripts', type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def snippet_key(paths) -> str:
    """Same key as the Whisper transcripts file (data_exploration/e3_alignment_whisper.py)."""
    return '|'.join(_clip_id(p) for p in paths)


def load_transcripts(path: Path) -> dict:
    transcripts = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                transcripts[record['key']] = record.get('transcript', '') or ''
    return transcripts


@th.no_grad()
def predict(model, texts, config, device, batch_size) -> np.ndarray:
    collator = TextTransformerCollator(model_card=config.model_card, tokenizer_args=config.tokenizer_args)
    preds = []
    for start in range(0, len(texts), batch_size):
        enc = collator(inputs=list(texts[start:start + batch_size]), context=None)
        logits = model({'inputs': enc['inputs'].to(device), 'input_mask': enc['input_mask'].to(device)})
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    args = parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    config = load_config(TransformerConfig, ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY,
                                                      task_name=TASK, tags={'anonymous', 'roberta'}))

    afc = MMUSEDFallacy(task_name=TASK, input_mode=InputMode.TEXT_ONLY, base_data_path=BASE_DATA_PATH).data
    test = afc[afc['fallacy'].notna() & afc['dialogue_id'].isin(SHARED_TASK_TEST_DIALOGUES)].reset_index(drop=True)
    labels = test['fallacy'].astype(int).to_numpy()
    transcripts = load_transcripts(args.transcripts)
    keys = test['snippet_paths'].map(snippet_key)
    missing = int((~keys.isin(transcripts.keys())).sum())
    if missing:
        raise RuntimeError(f'{missing} test snippets have no Whisper transcript in {args.transcripts}')
    whisper_text = [transcripts[k] for k in keys]
    n_empty = sum(1 for t in whisper_text if not t.strip())
    logger.info(f'{len(test)} test snippets, {n_empty} with an empty Whisper transcript')

    out = result_dir(SHARED_TASK_SPLIT, TASK, f'audio_only_cascade-whisper_{args.text_run}')
    out.mkdir(parents=True, exist_ok=True)
    scores = {'annotated': [], 'whisper': []}
    for seed in args.seeds:
        weights = result_dir(SHARED_TASK_SPLIT, TASK, args.text_run) / f'model_seed{seed}_fold0.pt'
        if not weights.exists():
            raise FileNotFoundError(f'{weights} not found: train it with --save-model (src.experiments.baseline)')
        model = Transformer(model_card=config.model_card, head=config.head, dropout_rate=config.dropout_rate,
                            is_transformer_trainable=False)
        model.load_state_dict(th.load(weights, map_location='cpu'))
        model.to(device).eval()

        pred_annotated = predict(model, test['snippet'].tolist(), config, device, args.batch_size)
        pred_whisper = predict(model, whisper_text, config, device, args.batch_size)
        scores['annotated'].append(macro_f1(labels, pred_annotated))
        scores['whisper'].append(macro_f1(labels, pred_whisper))
        logger.info(f'[seed={seed}] test macro-F1  annotated text {scores["annotated"][-1]:.4f} (sanity: must equal '
                    f'the text run)  |  Whisper transcript {scores["whisper"][-1]:.4f}  |  predictions changed by '
                    f'ASR: {np.mean(pred_annotated != pred_whisper):.1%}')
        np.savez(out / f'test_predictions_seed{seed}_fold0.npz', y_true=labels, y_pred=pred_whisper)

    whisper = np.array(scores['whisper'])
    np.save(out / 'metrics.npy', {'test': {'test_f1': scores['whisper'], 'avg_test_f1': (whisper.mean(), whisper.std()),
                                           'annotated_text_f1': scores['annotated']},
                                  'seeds': args.seeds, 'args': {k: str(v) for k, v in vars(args).items()}},
            allow_pickle=True)
    logger.info(f'{out.name}: Whisper test macro-F1 {whisper.mean():.4f} +- {whisper.std():.4f} over seeds {args.seeds}')


if __name__ == '__main__':
    main()
