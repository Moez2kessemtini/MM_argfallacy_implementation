"""Repository paths (data, results, feature cache); overridable with environment variables."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()

BASE_DATA_PATH = Path(os.environ.get('MAMKIT_DATA_PATH', REPO_ROOT / 'data'))
RESULTS_DIR = REPO_ROOT / 'results'
FEATURE_CACHE_DIR = Path(os.environ.get('MAMKIT_FEATURE_CACHE', REPO_ROOT / 'cache'))
