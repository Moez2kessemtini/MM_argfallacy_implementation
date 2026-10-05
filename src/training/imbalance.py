"""Class-imbalance strategies: configuration weights, training-set weights, focal loss, balanced sampler."""
from typing import Callable, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Sampler, WeightedRandomSampler

IMBALANCE_STRATEGIES = ('weighted', 'weighted_train', 'focal', 'sampler')
FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL = -alpha_y * (1 - p_y)^gamma * log(p_y)."""

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: Optional[th.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', None if alpha is None else th.as_tensor(alpha, dtype=th.float32))

    def forward(self, logits: th.Tensor, targets: th.Tensor) -> th.Tensor:
        log_p = F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - log_p.exp()) ** self.gamma) * log_p
        if self.alpha is None:
            return loss.mean()
        weights = self.alpha[targets]
        return (weights * loss).sum() / weights.sum()


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """N / (K * n_c); classes absent from `labels` get weight 0 (they never appear as targets)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = len(labels) / (num_classes * counts[present])
    return weights


def build_imbalance_strategy(name: str, config_loss_function: Callable[[], nn.Module],
                             train_labels: np.ndarray, num_classes: int):
    """Returns (loss_function factory, train sampler or None, description dict for metrics.npy)."""
    if name not in IMBALANCE_STRATEGIES:
        raise ValueError(f'imbalance strategy must be one of {IMBALANCE_STRATEGIES}, got {name!r}')
    sampler: Optional[Sampler] = None

    if name == 'weighted':
        loss_function = config_loss_function
        weights = getattr(config_loss_function(), 'weight', None)
        info = {'strategy': name, 'class_weights': None if weights is None else weights.tolist()}
    elif name == 'weighted_train':
        weights = th.tensor(inverse_frequency_weights(train_labels, num_classes), dtype=th.float32)
        loss_function = lambda: nn.CrossEntropyLoss(weight=weights)  # noqa: E731
        info = {'strategy': name, 'class_weights': weights.tolist()}
    elif name == 'focal':
        loss_function = lambda: FocalLoss(gamma=FOCAL_GAMMA)  # noqa: E731
        info = {'strategy': name, 'gamma': FOCAL_GAMMA}
    else:  # sampler
        class_weights = inverse_frequency_weights(train_labels, num_classes)
        sampler = WeightedRandomSampler(weights=th.tensor(class_weights[train_labels], dtype=th.double),
                                        num_samples=len(train_labels), replacement=True)
        loss_function = nn.CrossEntropyLoss
        info = {'strategy': name, 'sampling_weights': class_weights.tolist()}
    return loss_function, sampler, info
