"""Post-hoc analyses of the saved predictions: macro-F1 conventions and paired bootstrap table.

Usage:
  python -m src.evaluation.posthoc
"""
import argparse

import numpy as np

from src.evaluation.bootstrap import headline_f1, load_predictions
from src.paths import RESULTS_DIR

SPLIT = 'mm-argfallacy-2025'
AFC_CLASSES = 6

PAIRS = [
    # (task, reference A, compared B, label)
    ('afc', 'text_only_roberta', 'text_audio_roberta_wavlm', 'texte+audio vs texte (gelé)'),
    ('afd', 'text_only_roberta', 'text_audio_roberta_wavlm', 'texte+audio vs texte (gelé)'),
    ('afc', 'text_only_roberta', 'text_only_roberta_ft', 'fine-tuné vs gelé'),
    ('afd', 'text_only_roberta', 'text_only_roberta_ft', 'fine-tuné vs gelé'),
    ('afc', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-weighted_train', 'poids recalculés vs poids config (ft)'),
    ('afd', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-weighted_train', 'poids recalculés vs poids config (ft)'),
    ('afc', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-focal', 'focal vs poids config (ft)'),
    ('afd', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-focal', 'focal vs poids config (ft)'),
    ('afc', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-sampler', 'sampler vs poids config (ft)'),
    ('afd', 'text_only_roberta_ft', 'text_only_roberta_ft_imb-sampler', 'sampler vs poids config (ft)'),
    ('afc', 'text_only_roberta_ft', 'text_audio_roberta_wavlm_ft', 'texte+audio vs texte (ft)'),
    ('afd', 'text_only_roberta_ft', 'text_audio_roberta_wavlm_ft', 'texte+audio vs texte (ft)'),
    ('afc', 'text_audio_roberta_wavlm', 'text_audio_roberta_wavlm_fusion-early', 'fusion précoce vs concaténation'),
    ('afc', 'text_audio_roberta_wavlm', 'text_audio_roberta_wavlm_fusion-late_lr0.0001', 'fusion tardive (lr choisi) vs concaténation'),
    ('afc', 'text_audio_roberta_wavlm', 'text_audio_roberta_wavlm_fusion-crossattn_lr5e-05', 'attention croisée (lr choisi) vs concaténation'),
    ('afc', 'text_audio_roberta_wavlm', 'text_audio_roberta_wavlm_fusion-intermediate_lr0.0001', 'intermédiaire (lr choisi) vs concaténation'),
    ('afc', 'text_audio_roberta_wavlm', 'text_audio_roberta_wavlm_fusion-selfattn_lr0.0001', 'auto-attention (lr choisi) vs concaténation'),
    ('afc', 'text_audio_roberta_wavlm_fusion-textonly_lr0.0001', 'text_audio_roberta_wavlm_fusion-late_lr0.0001', 'fusion tardive vs sa branche texte (lr 1e-4)'),
    ('afc', 'text_only_roberta_ctx1', 'text_only_roberta_ctx1-tgtpool', 'ctx1 : moyenne sur la cible vs sur la paire'),
    ('afc', 'text_only_roberta_ctx2', 'text_only_roberta_ctx2-tgtpool', 'ctx2 : moyenne sur la cible vs sur la paire'),
    ('afc', 'text_only_roberta', 'text_only_roberta_ctx1-tgtpool', 'ctx1 moyenne sur la cible vs phrase seule'),
    ('afc', 'text_only_roberta', 'text_only_roberta_ctx2-tgtpool', 'ctx2 moyenne sur la cible vs phrase seule'),
    ('afd', 'text_only_roberta', 'text_only_roberta_ctx1-tgtpool', 'ctx1 moyenne sur la cible vs phrase seule'),
    ('afd', 'text_only_roberta', 'text_only_roberta_ctx2-tgtpool', 'ctx2 moyenne sur la cible vs phrase seule'),
    ('afc', 'audio_only_wavlm_raw_readout-linear', 'audio_only_wavlm_comodo-generic_tt0.02_readout-linear', 'distillation enseignant générique vs raw'),
    ('afc', 'audio_only_wavlm_comodo-generic_tt0.02_readout-linear', 'audio_only_wavlm_comodo-finetuned_tt0.02_readout-linear', 'enseignant fine-tuné vs générique'),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def macro_conventions(y_true, y_pred):
    k = AFC_CLASSES
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    gold = cm.sum(1) > 0
    return {'present': f1[denom > 0].mean(), 'gold': f1[gold].mean(), 'all6': f1.mean()}


def convention_table():
    root = RESULTS_DIR / 'mmused-fallacy' / SPLIT / 'afc'
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            preds = load_predictions('afc', run_dir.name, SPLIT)
        except FileNotFoundError:
            continue
        scores = [macro_conventions(*preds[s]) for s in sorted(preds)]
        mean = {c: np.mean([s[c] for s in scores]) for c in ('present', 'gold', 'all6')}
        n_slippery = np.mean([(preds[s][1] == 4).sum() for s in preds])
        rows.append((run_dir.name, len(preds), mean['present'], mean['gold'], mean['all6'], n_slippery))
    lines = ['| run (AFC) | seeds | F1 present | F1 gold (5 classes) | F1 6 classes | Slippery Slope predicted |',
             '|---|---|---|---|---|---|']
    lines += [f'| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.1f} |' for r in rows]
    return '\n'.join(lines)


def bootstrap(task, a, b, n_boot, seed):
    preds_a, preds_b = load_predictions(task, a, SPLIT), load_predictions(task, b, SPLIT)
    seeds = sorted(set(preds_a) & set(preds_b))
    if not seeds:
        raise FileNotFoundError('no common seed')
    y_true = preds_a[seeds[0]][0]
    for s in seeds:
        if not (np.array_equal(preds_a[s][0], y_true) and np.array_equal(preds_b[s][0], y_true)):
            raise ValueError(f'seed {s}: different test labels')

    def score(preds, idx):
        return np.mean([headline_f1(y_true[idx], preds[s][1][idx], task) for s in seeds])

    full = np.arange(len(y_true))
    observed = score(preds_b, full) - score(preds_a, full)
    rng = np.random.default_rng(seed)
    diffs = np.array([score(preds_b, i) - score(preds_a, i)
                      for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))])
    low, high = np.percentile(diffs, [2.5, 97.5])
    p_two = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return score(preds_a, full), score(preds_b, full), observed, low, high, p_two, len(seeds)


def bootstrap_table(n_boot, seed):
    lines = ['| task | comparison (B vs A) | A | B | B - A [95% CI] | p (two-sided) | seeds |', '|---|---|---|---|---|---|---|']
    for task, a, b, label in PAIRS:
        try:
            fa, fb, d, lo, hi, p, n = bootstrap(task, a, b, n_boot, seed)
        except (FileNotFoundError, ValueError) as e:
            print(f'  skipped {task} {b} vs {a}: {e}')
            continue
        lines.append(f'| {task.upper()} | {label} | {fa:.4f} | {fb:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {n} |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    conv = convention_table()
    print('## Macro-F1 convention (AFC test, mean over seeds)\n')
    print(conv)
    boot = bootstrap_table(args.n_boot, args.seed)
    print('\n## Paired bootstrap\n')
    print(boot)
    (RESULTS_DIR / 'extra_analyses.md').write_text(
        '## Macro-F1 convention (AFC test, mean over seeds)\n\n' + conv + '\n\n## Paired bootstrap\n\n' + boot + '\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
