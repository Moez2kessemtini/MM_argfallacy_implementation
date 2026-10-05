"""Reproduced vs. published Baseline Transformer scores (Table 4).

Usage:
  python -m src.evaluation.aggregate
"""
import numpy as np
import pandas as pd

from src.paths import RESULTS_DIR
from src.training.loop import SHARED_TASK_SPLIT

PAPER_TABLE4 = {
    ('afc', 'text'):       {'Baseline BiLSTM': 0.4721, 'Baseline Transformer': 0.3925},
    ('afc', 'audio'):      {'Baseline BiLSTM': 0.1582, 'Baseline Transformer': 0.0643},
    ('afc', 'text_audio'): {'Baseline BiLSTM': 0.2191, 'Baseline Transformer': 0.3816},
    ('afd', 'text'):       {'Baseline BiLSTM': 0.2462, 'Baseline Transformer': 0.2770},
    ('afd', 'audio'):      {'Baseline BiLSTM': 0.0000, 'Baseline Transformer': 0.0000},
    ('afd', 'text_audio'): {'Baseline BiLSTM': 0.2337, 'Baseline Transformer': 0.2848},
}

RESULT_DIR_NAME = {
    'text': 'text_only_roberta',
    'audio': 'audio_only_transformer_wavlm',
    'text_audio': 'text_audio_roberta_wavlm',
}


def load_metrics(task: str, modality: str):
    # Table 4 is only comparable to the official shared-task split.
    path = RESULTS_DIR / 'mmused-fallacy' / SHARED_TASK_SPLIT / task / RESULT_DIR_NAME[modality] / 'metrics.npy'
    if not path.exists():
        return None
    metrics = np.load(path, allow_pickle=True).item()
    avg, std = metrics['test']['avg_test_f1']
    return float(avg), float(std)


def build_table() -> pd.DataFrame:
    rows = []
    for task in ['afc', 'afd']:
        for modality in ['text', 'audio', 'text_audio']:
            reproduced = load_metrics(task, modality)
            paper_ref = PAPER_TABLE4[(task, modality)]['Baseline Transformer']
            rows.append({
                'task': task.upper(),
                'modality': modality,
                'paper_baseline_transformer_f1': paper_ref,
                'reproduced_f1_mean': reproduced[0] if reproduced else None,
                'reproduced_f1_std': reproduced[1] if reproduced else None,
                'delta': (reproduced[0] - paper_ref) if reproduced else None,
                'status': 'done' if reproduced else 'not run yet',
            })
    return pd.DataFrame(rows)


def print_markdown(df: pd.DataFrame) -> None:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '-'
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    headers = list(df.columns)
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in df.iterrows():
        print('| ' + ' | '.join(fmt(row[h]) for h in headers) + ' |')


if __name__ == '__main__':
    table = build_table()
    print_markdown(table)
    out_csv = RESULTS_DIR / 'table4_reproduction.csv'
    table.to_csv(out_csv, index=False)
    print(f'\nSaved to {out_csv}')
