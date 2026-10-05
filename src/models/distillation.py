"""Text-to-audio distillation: frozen text teacher, audio student and queue-based distillation loss."""
import math
from typing import Optional

import torch as th
import torch.nn.functional as F
from torch import nn

from mamkit.models.audio import TransformerEncoder
from mamkit.models.text import Transformer


def projection_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, output_dim))


class QueueDistillationLoss(nn.Module):
    """Cross-entropy between teacher and student similarity distributions over a FIFO queue."""

    def __init__(self, initial_queue: th.Tensor, teacher_temp: float = 0.1, student_temp: float = 0.05):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.register_buffer('queue', F.normalize(initial_queue.detach().float(), dim=1))

    def forward(self, z_student: th.Tensor, z_teacher: th.Tensor) -> th.Tensor:
        """z_student, z_teacher: [B, D], L2-normalised."""
        bank = th.cat([self.queue, z_teacher.detach()])
        target = F.softmax(z_teacher.detach() @ bank.T / self.teacher_temp, dim=1)
        log_pred = F.log_softmax(z_student @ bank.T / self.student_temp, dim=1)
        loss = -(target * log_pred).sum(dim=1).mean()

        with th.no_grad():
            entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=1).mean()
            own = len(self.queue) + th.arange(len(z_teacher), device=target.device)
            self.last_stats = {'loss': loss.item(), 'target_entropy': entropy.item(),
                               'kl': loss.item() - entropy.item(), 'log_bank_size': math.log(bank.shape[0]),
                               'target_self_prob': target[th.arange(len(z_teacher)), own].mean().item()}

        self.queue = bank[len(z_teacher):].detach()  # FIFO: drop the oldest batch
        return loss


class AudioStudent(nn.Module):
    """Audio-only baseline encoder without its head, followed by a projector used for distillation."""

    def __init__(self, audio_config, proj_hidden_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = TransformerEncoder(embedding_dim=audio_config.embedding_dim,
                                          encoder=audio_config.encoder, head=nn.Identity,
                                          dropout_rate=audio_config.dropout_rate)
        self.projector = projection_head(audio_config.embedding_dim, proj_hidden_dim, proj_dim)

    def represent(self, inputs: dict) -> th.Tensor:
        """Pooled audio representation [B, embedding_dim] -- what the classifiers read out."""
        return self.encoder(inputs)

    def forward(self, inputs: dict) -> th.Tensor:
        return F.normalize(self.projector(self.represent(inputs)), dim=1)


class TextTeacher(nn.Module):
    """Frozen mamkit text Transformer (RoBERTa, mean-pooled tokens) + frozen random projector."""

    def __init__(self, text_config, proj_hidden_dim: int, proj_dim: int, state_dict: Optional[dict] = None):
        super().__init__()
        self.encoder = Transformer(model_card=text_config.model_card, head=text_config.head,
                                   dropout_rate=text_config.dropout_rate, is_transformer_trainable=False)
        if state_dict is not None:
            self.encoder.load_state_dict(state_dict)
        self.encoder.head = nn.Identity()  # keep the mean-pooled sentence representation
        self.projector = projection_head(self.encoder.model_config.hidden_size, proj_hidden_dim, proj_dim)
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):  # always frozen, dropout off
        return super().train(False)

    def set_standardization(self, features: th.Tensor):
        """features: [N, hidden] pooled features of the TRAINING sentences only."""
        self.feature_mean = features.mean(dim=0).to(self.projector[0].weight.device)
        self.feature_std = features.std(dim=0).clamp_min(1e-6).to(self.projector[0].weight.device)

    @th.no_grad()
    def pooled(self, inputs: dict) -> th.Tensor:
        return self.encoder(inputs)

    @th.no_grad()
    def project(self, pooled: th.Tensor) -> th.Tensor:
        if self.feature_mean is not None:
            pooled = (pooled - self.feature_mean) / self.feature_std
        return F.normalize(self.projector(pooled), dim=1)

    def forward(self, inputs: dict) -> th.Tensor:
        return self.project(self.pooled(inputs))
