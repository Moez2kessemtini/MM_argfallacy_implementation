"""Shared helpers of the data exploration scripts."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # headless server: render to files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mamkit.data.datasets import InputMode, MMUSEDFallacy

from src.paths import BASE_DATA_PATH
from src.training.loop import SHARED_TASK_TEST_DIALOGUES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('data_exploration')

EXPLORATION_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPLORATION_DIR / 'outputs'
DATASET_PKL = BASE_DATA_PATH / 'MMUSED-fallacy' / 'dataset.pkl'

# Label ids as mapped by MMUSEDFallacy.data
AFC_LABELS = ['Appeal to Emotion', 'Appeal to Authority', 'Ad Hominem', 'False Cause', 'Slippery Slope', 'Slogans']
AFD_LABELS = ['No fallacy', 'Fallacy']
LABELS = {'afc': AFC_LABELS, 'afd': AFD_LABELS}

# train = MM-USED-Fallacy debates (training + validation pool), test = the two 2024 debates
SPLITS = ('train', 'test')
SPLIT_COLORS = {'train': '#4C72B0', 'test': '#DD8452'}
CLASS_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def load_task(task: str, context_window: int = 3) -> pd.DataFrame:
    """MM-USED-Fallacy samples of a task with label, text, split and year columns."""
    if context_window < 1:
        raise ValueError('context_window must be >= 1')
    loader = MMUSEDFallacy(task_name=task, input_mode=InputMode.TEXT_AUDIO, base_data_path=BASE_DATA_PATH,
                           context_window=context_window)
    df = loader.data.copy()
    if task == 'afc':
        df = df[df['fallacy'].notna()].copy()
        df['label'] = df['fallacy'].astype(int)
        df['text'] = df['snippet']
        df['context_text'] = df['dialogue']
    else:
        df['label'] = df['label'].astype(int)
        df['text'] = df['sentence']
        df['context_text'] = df['context']
    df['label_name'] = [LABELS[task][i] for i in df['label']]
    df['split'] = np.where(df['dialogue_id'].isin(SHARED_TASK_TEST_DIALOGUES), 'test', 'train')
    df['year'] = df['dialogue_id'].str.split('_').str[-1].astype(int)
    return df.reset_index(drop=True)


def debate_order(df: pd.DataFrame) -> list:
    """Debate ids sorted chronologically (year, then debate number)."""
    ids = df['dialogue_id'].unique()
    return sorted(ids, key=lambda d: (int(d.split('_')[-1]), int(d.split('_')[0])))


def to_markdown(df: pd.DataFrame, float_fmt: str = '{:.3f}') -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return '-' if np.isnan(value) else float_fmt.format(value)
        return str(value)

    header = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
    rule = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, rule, *rows])


class Outputs:
    """Figures / tables / summary.md of one exploration step."""

    def __init__(self, step: str):
        self.dir = OUTPUT_ROOT / step
        self.figures = self.dir / 'figures'
        self.tables = self.dir / 'tables'
        self.figures.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        self._summary = [f'# {step}\n']

    def figure(self, fig, name: str):
        path = self.figures / f'{name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'figure -> {path.relative_to(EXPLORATION_DIR)}')

    def table(self, df: pd.DataFrame, name: str, index: bool = False):
        path = self.tables / f'{name}.csv'
        df.to_csv(path, index=index)
        logger.info(f'table  -> {path.relative_to(EXPLORATION_DIR)}')

    def section(self, title: str):
        self._summary.append(f'\n## {title}\n')

    def text(self, line: str):
        self._summary.append(line)

    def md_table(self, df: pd.DataFrame, float_fmt: str = '{:.3f}'):
        self._summary.append(to_markdown(df, float_fmt) + '\n')

    def write_summary(self):
        path = self.dir / 'summary.md'
        path.write_text('\n'.join(self._summary) + '\n', encoding='utf-8')
        logger.info(f'summary -> {path.relative_to(EXPLORATION_DIR)}')
