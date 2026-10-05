"""Per-class scores and confusion matrices from the saved test predictions.

Usage:
  python -m src.evaluation.per_class --task afc --runs ctx
"""
import argparse
import re

import numpy as np
import pandas as pd

from src.evaluation.aggregate import print_markdown
from src.paths import RESULTS_DIR
from src.training.loop import SHARED_TASK_SPLIT

# Label ids as mapped by mamkit's MMUSEDFallacy.data
CLASS_NAMES = {
    'afc': ['AppealtoEmotion', 'AppealtoAuthority', 'AdHominem', 'FalseCause', 'Slipperyslope', 'Slogans'],
    'afd': ['NoFallacy', 'Fallacy'],
}

PRED_FILE_RE = re.compile(r'test_predictions_seed(\d+)_fold(\d+)\.npz')


def confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm  # rows = gold, cols = predicted


def per_class_scores(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    present = (tp + fp + fn) > 0
    return precision, recall, f1, present


def headline_f1(task, f1, present):
    if task == 'afd':
        return f1[1]  # binary F1 = F1 of the positive (fallacy) class
    return f1[present].mean()  # macro over classes present in gold or predictions


def find_run_dirs(task, split, name_filters=None):
    task_dir = RESULTS_DIR / 'mmused-fallacy' / split / task
    if not task_dir.exists():
        return []
    run_dirs = [d for d in sorted(task_dir.iterdir())
                if d.is_dir() and any(d.glob('test_predictions_seed*_fold*.npz'))]
    if name_filters:
        run_dirs = [d for d in run_dirs if any(f in d.name for f in name_filters)]
    return run_dirs


def load_runs(run_dir):
    runs = []
    for path in sorted(run_dir.glob('test_predictions_seed*_fold*.npz')):
        seed, fold = map(int, PRED_FILE_RE.match(path.name).groups())
        data = np.load(path)
        runs.append({'seed': seed, 'fold': fold,
                     'y_true': data['y_true'].astype(int), 'y_pred': data['y_pred'].astype(int)})

    logged = None
    metrics_path = run_dir / 'metrics.npy'
    if metrics_path.exists():
        logged = np.load(metrics_path, allow_pickle=True).item()['test'].get('test_f1')
    return runs, logged


def analyze(task, run_dir):
    runs, logged = load_runs(run_dir)

    names = CLASS_NAMES[task]
    k = len(names)
    per_run, cms = [], []
    for run in runs:
        cm = confusion(run['y_true'], run['y_pred'], k)
        precision, recall, f1, present = per_class_scores(cm)
        per_run.append({'precision': precision, 'recall': recall, 'f1': f1,
                        'headline': headline_f1(task, f1, present)})
        cms.append(cm)

    print(f'\n## {task.upper()} / {run_dir.name}  ({len(runs)} run(s): '
          f'seeds {sorted({r["seed"] for r in runs})})')

    recomputed = [r['headline'] for r in per_run]
    if logged is not None and len(logged) == len(recomputed):
        diffs = np.abs(np.sort(np.array(logged, dtype=float)) - np.sort(recomputed))
        status = 'OK' if diffs.max() < 1e-4 else f'MISMATCH (max diff {diffs.max():.4f})'
        print(f'Headline F1 recomputed vs logged: {status}')
    headline = np.array(recomputed)
    print(f'Headline F1 ({"binary" if task == "afd" else "macro"}): '
          f'{headline.mean():.4f} +- {headline.std():.4f}')

    gold_counts = np.bincount(runs[0]['y_true'], minlength=k)
    pred_counts = np.mean([np.bincount(r['y_pred'], minlength=k) for r in runs], axis=0)
    stack = {key: np.stack([r[key] for r in per_run]) for key in ('precision', 'recall', 'f1')}
    table = pd.DataFrame({
        'class': names,
        'gold_n': gold_counts,
        'pred_n_mean': pred_counts.round(1),
        'precision': [f'{m:.3f} +- {s:.3f}' for m, s in zip(stack['precision'].mean(0), stack['precision'].std(0))],
        'recall': [f'{m:.3f} +- {s:.3f}' for m, s in zip(stack['recall'].mean(0), stack['recall'].std(0))],
        'f1': [f'{m:.3f} +- {s:.3f}' for m, s in zip(stack['f1'].mean(0), stack['f1'].std(0))],
    })
    print_markdown(table)

    n_pred_classes = int((pred_counts > 0).sum())
    if n_pred_classes <= 1:
        only = names[int(pred_counts.argmax())]
        print(f'-> Degenerate classifier: always predicts {only!r}.')

    cm_sum = pd.DataFrame(np.sum(cms, axis=0), columns=[f'pred:{n}' for n in names])
    cm_sum.insert(0, 'gold', names)
    print('\nConfusion matrix (summed over runs, rows = gold):')
    print_markdown(cm_sum)

    out = run_dir / 'per_class_analysis.csv'
    table.to_csv(out, index=False)
    return {'task': task.upper(), 'run': run_dir.name, 'n_runs': len(runs),
            'f1_mean': headline.mean(), 'f1_std': headline.std(),
            'n_predicted_classes': n_pred_classes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['afc', 'afd'], nargs='+', default=['afc', 'afd'])
    parser.add_argument('--runs', nargs='+', default=None,
                        help='Only analyze run directories whose name contains one of these strings')
    parser.add_argument('--split', default=SHARED_TASK_SPLIT)
    args = parser.parse_args()

    summary = [analyze(task, run_dir) for task in args.task
               for run_dir in find_run_dirs(task, args.split, args.runs)]
    if not summary:
        print('No run directory with saved test predictions found.')
    else:
        print('\n## Summary')
        print_markdown(pd.DataFrame(summary))


if __name__ == '__main__':
    main()
