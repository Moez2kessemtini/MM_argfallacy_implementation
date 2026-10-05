"""Shared helpers: macro F1 and deterministic GPU kernels."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # must precede the first CUDA call

import numpy as np
import torch as th
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred) -> float:
    """Macro F1 over the classes present in the gold labels or the predictions."""
    return f1_score(y_true, y_pred, labels=np.union1d(y_true, y_pred), average='macro', zero_division=0)


def enable_determinism():
    """Deterministic cuDNN/cuBLAS kernels (reduces, but does not remove, run-to-run variation)."""
    th.use_deterministic_algorithms(True, warn_only=True)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
