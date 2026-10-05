"""Experiment variants on top of the baselines; each option is reflected in the run name."""
from dataclasses import dataclass
from typing import Optional

from src.models.fusion import FUSION_TYPES
from src.training.imbalance import IMBALANCE_STRATEGIES

FINETUNE_LR = 1e-5


@dataclass(frozen=True)
class ExperimentOptions:
    fusion: str = 'concat'
    finetune: bool = False
    lr: Optional[float] = None
    imbalance: str = 'weighted'
    context: int = 0
    context_pooling: str = 'all'

    def __post_init__(self):
        if self.context < 0:
            raise ValueError(f'context must be >= 0, got {self.context}')
        if self.context_pooling not in ('all', 'target'):
            raise ValueError(f"context_pooling must be 'all' or 'target', got {self.context_pooling!r}")
        if self.fusion not in FUSION_TYPES:
            raise ValueError(f'fusion must be one of {FUSION_TYPES}, got {self.fusion!r}')
        if self.imbalance not in IMBALANCE_STRATEGIES:
            raise ValueError(f'imbalance must be one of {IMBALANCE_STRATEGIES}, got {self.imbalance!r}')
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f'lr must be > 0, got {self.lr}')

    def run_name(self, base: str) -> str:
        name = base
        if self.fusion != 'concat':
            name += f'_fusion-{self.fusion}'
        if self.finetune:
            name += '_ft'
        if self.lr is not None:
            name += f'_lr{self.lr:g}'
        if self.imbalance != 'weighted':
            name += f'_imb-{self.imbalance}'
        if self.context:
            name += f'_ctx{self.context}'
            if self.context_pooling == 'target':
                name += '-tgtpool'
        return name

    def text_encoder_trainable(self, config_value: bool) -> bool:
        return True if self.finetune else config_value

    def optimizer_args(self, config_args: dict) -> dict:
        args = dict(config_args)
        if self.lr is not None:
            args['lr'] = self.lr
        elif self.finetune:
            args['lr'] = FINETUNE_LR
        return args


BASELINE = ExperimentOptions()
