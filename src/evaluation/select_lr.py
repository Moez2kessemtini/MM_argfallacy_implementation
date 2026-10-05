"""Learning-rate selection of the fusion models on the validation F1.

Usage:
  python -m src.evaluation.select_lr --task afc --lrs 1e-4 5e-5
"""
import argparse

import numpy as np

from src.paths import RESULTS_DIR

FUSIONS = ('early', 'intermediate', 'late', 'selfattn', 'crossattn', 'textonly')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=['afc', 'afd'], required=True)
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-4, 5e-5])
    parser.add_argument('--default-lr', type=float, default=1e-3, help='lr of the runs without suffix')
    parser.add_argument('--split', default='mm-argfallacy-2025')
    return parser.parse_args()


def load(run_dir):
    path = run_dir / 'metrics.npy'
    if not path.exists():
        return None
    m = np.load(path, allow_pickle=True).item()
    val = np.asarray(m['validation']['test_f1'], dtype=float)
    test = np.asarray(m['test']['test_f1'], dtype=float)
    return val, test


def main():
    args = parse_args()
    root = RESULTS_DIR / 'mmused-fallacy' / args.split / args.task
    print(f'{args.task.upper()}: lr chosen on mean validation F1 (test never used for the choice)\n')
    print(f'{"fusion":<13}{"lr":>8}{"val F1":>10}{"test F1 (mean +- std)":>26}   per seed')
    summary = []
    for fusion in FUSIONS:
        base = f'text_audio_roberta_wavlm_fusion-{fusion}'
        candidates = [(args.default_lr, root / base)] + [(lr, root / f'{base}_lr{lr:g}') for lr in args.lrs]
        found = [(lr, r) for lr, r in ((lr, load(d)) for lr, d in candidates) if r is not None]
        if not found:
            continue
        for lr, (val, test) in found:
            print(f'{fusion:<13}{lr:>8g}{val.mean():>10.4f}{test.mean():>14.4f} +- {test.std():.4f}   '
                  f'{np.round(test, 4).tolist()}')
        lr, (val, test) = max(found, key=lambda item: item[1][0].mean())
        summary.append((fusion, lr, val.mean(), test.mean(), test.std(), len(found)))
        print()
    print('Selected:')
    for fusion, lr, val, mean, std, n in summary:
        print(f'  {fusion:<13} lr {lr:g}  (val {val:.4f}, {n} candidates)  ->  test {mean:.4f} +- {std:.4f}')


if __name__ == '__main__':
    main()
