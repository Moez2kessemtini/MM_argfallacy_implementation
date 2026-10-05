"""Paired bootstrap comparison of two runs on the same test samples.

Usage:
  python -m src.evaluation.bootstrap --task afc --a <run A> --b <run B>
"""
import argparse
import re

import numpy as np

from src.paths import RESULTS_DIR

NUM_CLASSES = {'afc': 6, 'afd': 2}
PRED_RE = re.compile(r'test_predictions_seed(\d+)_fold0\.npz')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=list(NUM_CLASSES), required=True)
    parser.add_argument('--a', required=True, help='reference run directory name')
    parser.add_argument('--b', required=True, help='compared run directory name')
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', default='mm-argfallacy-2025')
    return parser.parse_args()


def load_predictions(task: str, run: str, split: str) -> dict:
    run_dir = RESULTS_DIR / 'mmused-fallacy' / split / task / run
    preds = {}
    for path in run_dir.glob('test_predictions_seed*_fold0.npz'):
        data = np.load(path)
        preds[int(PRED_RE.match(path.name).group(1))] = (data['y_true'].astype(int), data['y_pred'].astype(int))
    if not preds:
        raise FileNotFoundError(f'no test predictions in {run_dir}')
    return preds


def headline_f1(y_true, y_pred, task: str) -> float:
    """Macro F1 over classes present in gold or predictions (AFC) / binary F1 (AFD)."""
    k = NUM_CLASSES[task]
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1[1] if task == 'afd' else f1[denom > 0].mean()


def main():
    args = parse_args()
    preds_a = load_predictions(args.task, args.a, args.split)
    preds_b = load_predictions(args.task, args.b, args.split)
    seeds = sorted(set(preds_a) & set(preds_b))
    if not seeds:
        raise ValueError(f'no common seed: {args.a} has {sorted(preds_a)}, {args.b} has {sorted(preds_b)}')
    y_true = preds_a[seeds[0]][0]
    for s in seeds:
        if not (np.array_equal(preds_a[s][0], y_true) and np.array_equal(preds_b[s][0], y_true)):
            raise ValueError(f'seed {s}: test labels differ between the runs -- not the same test samples/order')

    def score(preds, idx):
        return np.mean([headline_f1(y_true[idx], preds[s][1][idx], args.task) for s in seeds])

    full = np.arange(len(y_true))
    observed = score(preds_b, full) - score(preds_a, full)
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[i] = score(preds_b, idx) - score(preds_a, idx)
    low, high = np.percentile(diffs, [2.5, 97.5])
    p_one = float(np.mean(diffs <= 0))
    p_two = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))

    metric = 'binary F1' if args.task == 'afd' else 'macro F1'
    print(f'{args.task.upper()} test, {len(y_true)} samples, seeds {seeds}, {args.n_boot} paired resamples ({metric})')
    for name, preds in (('A', preds_a), ('B', preds_b)):
        per_seed = [headline_f1(y_true, preds[s][1], args.task) for s in seeds]
        run = args.a if name == 'A' else args.b
        print(f'  {name} {run}: mean {np.mean(per_seed):.4f}  per seed {np.round(per_seed, 4).tolist()}')
    print(f'  B - A = {observed:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]   '
          f'p(B <= A) = {p_one:.4f}   two-sided p = {p_two:.4f}')


if __name__ == '__main__':
    main()
