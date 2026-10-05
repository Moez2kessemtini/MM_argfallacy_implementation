"""Download MM-USED-Fallacy through MAMKit, build the audio clips and fix the label encoding.

Usage:
  python -m src.data.prepare
"""
import importlib
import logging
import shutil
import sys

import pandas as pd

from mamkit.data.datasets import InputMode, MMUSEDFallacy

from src.paths import BASE_DATA_PATH
from src.training.loop import SHARED_TASK_TEST_DIALOGUES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PKL = BASE_DATA_PATH / 'MMUSED-fallacy' / 'dataset.pkl'


def _alias_numpy2_modules():
    """Map numpy._core.* to numpy.core.* so NumPy 1.x can unpickle NumPy 2.x objects."""
    import numpy.core
    sys.modules.setdefault('numpy._core', numpy.core)
    for name in ('numeric', 'multiarray', '_multiarray_umath', 'umath', 'fromnumeric',
                 'records', 'defchararray', '_dtype_ctypes', '_internal'):
        try:
            sys.modules.setdefault(f'numpy._core.{name}', importlib.import_module(f'numpy.core.{name}'))
        except ImportError:
            pass


def convert_numpy2_pickle_if_needed():
    if not DATASET_PKL.exists():
        return
    try:
        pd.read_pickle(DATASET_PKL)
        return
    except ModuleNotFoundError as e:
        if 'numpy._core' not in str(e):
            raise
    logger.info(f'{DATASET_PKL.name} was pickled with NumPy 2.x, converting for NumPy 1.x...')
    _alias_numpy2_modules()
    df = pd.read_pickle(DATASET_PKL)
    backup = DATASET_PKL.with_name(DATASET_PKL.name + '.numpy2.bak')
    if not backup.exists():
        shutil.copy2(DATASET_PKL, backup)
    df.to_pickle(DATASET_PKL)
    logger.info(f'Converted ({len(df)} rows); original kept at {backup.name}')


def normalize_missing_fallacy_labels():
    """Store missing fallacy labels as None so that MAMKit labels non-fallacious AFD sentences 0."""
    if not DATASET_PKL.exists():
        return
    df = pd.read_pickle(DATASET_PKL)
    n_nan = sum(1 for v in df['fallacy'] if v is not None and pd.isna(v))
    if n_nan == 0:
        return
    df['fallacy'] = pd.Series([None if pd.isna(v) else v for v in df['fallacy']],
                              index=df.index, dtype=object)
    df.to_pickle(DATASET_PKL)
    logger.info(f'Rewrote {n_nan} missing fallacy labels from NaN to None')


def main():
    logger.info(f'Preparing MM-USED-Fallacy under {BASE_DATA_PATH}')
    convert_numpy2_pickle_if_needed()
    normalize_missing_fallacy_labels()
    # Same on-disk data for both tasks; building one loader triggers download + clip generation.
    loader = MMUSEDFallacy(task_name='afc', input_mode=InputMode.TEXT_AUDIO, base_data_path=BASE_DATA_PATH)
    clips_path = loader.clips_path
    n_clips = sum(1 for _ in clips_path.rglob('*.wav'))
    logger.info(f'Done: {n_clips} audio clips in {clips_path}')

    label_col = {'afc': 'fallacy', 'afd': 'label'}
    for task_name in ('afc', 'afd'):
        data = MMUSEDFallacy(task_name=task_name, input_mode=InputMode.TEXT_ONLY,
                             base_data_path=BASE_DATA_PATH).data
        is_test = data['dialogue_id'].isin(SHARED_TASK_TEST_DIALOGUES)
        for name, part in (('train+val', data[~is_test]), ('test', data[is_test])):
            if task_name == 'afc':
                part = part[part['fallacy'].notna()]  # same filter as src.training.loop.shared_task_split
            counts = part[label_col[task_name]].value_counts(dropna=False).to_dict()
            logger.info(f'[{task_name}] {name}: {len(part)} samples, '
                        f'{part.dialogue_id.nunique()} debates, labels={counts}')


if __name__ == '__main__':
    main()
