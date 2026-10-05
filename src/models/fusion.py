"""Text-audio fusion models: early, intermediate, late, joint self-attention, cross-attention (+ text-only control)."""
import torch as th
from torch import nn

FUSION_TYPES = ('concat', 'early', 'intermediate', 'late', 'selfattn', 'crossattn', 'textonly')


def masked_mean(x: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """x: [bs, L, d], mask: [bs, L] (1 = real, 0 = padding) -> [bs, d]."""
    mask = mask.to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def padding_mask(mask: th.Tensor) -> th.Tensor:
    """mamkit masks are 1 for real positions; torch attention wants True for padding."""
    return mask == 0


def classification_head(d_in: int, num_classes: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))


class FrozenTextEncoder(nn.Module):
    """RoBERTa (or any HF encoder), frozen exactly as in the baseline -> token embeddings."""

    def __init__(self, model_card: str, trainable: bool = False):
        super().__init__()
        from transformers import AutoModel  # local import: keeps this module importable without HF
        self.model = AutoModel.from_pretrained(model_card)
        self.trainable = trainable
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        with th.set_grad_enabled(self.trainable and th.is_grad_enabled()):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class UnimodalEncoder(nn.Module):
    """Projection to d_model + one Transformer layer + masked mean pooling."""

    def __init__(self, d_in: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(d_in, d_model)
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                                                dropout=dropout, batch_first=True)

    def forward(self, x, mask):
        hidden = self.layer(self.projection(x), src_key_padding_mask=padding_mask(mask))
        return masked_mean(hidden, mask)


class CrossAttentionBlock(nn.Module):
    """Queries from one modality attend to the other modality (+ residual, LayerNorm, FFN)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d_model, d_model))
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, keys_mask):
        attended, _ = self.attention(queries, keys, keys, key_padding_mask=padding_mask(keys_mask))
        hidden = self.norm_1(queries + self.dropout(attended))
        return self.norm_2(hidden + self.dropout(self.ffn(hidden)))


class TextAudioFusionModel(nn.Module):
    """One model class for all fusion types except `concat` (mamkit baseline, see runner)."""

    def __init__(
            self,
            fusion: str,
            model_card: str,
            num_classes: int,
            audio_embedding_dim: int = 768,
            d_model: int = 256,
            n_heads: int = 4,
            text_dropout_rate: float = 0.2,
            audio_dropout_rate: float = 0.2,
            is_transformer_trainable: bool = False,
    ):
        super().__init__()
        if fusion not in FUSION_TYPES or fusion == 'concat':
            raise ValueError(f"fusion must be one of {FUSION_TYPES[1:]} here, got {fusion!r} "
                             f"('concat' is the mamkit baseline, built by src.training.runner)")
        self.fusion = fusion

        self.text_encoder = FrozenTextEncoder(model_card, trainable=is_transformer_trainable)
        d_text, d_audio = self.text_encoder.hidden_size, audio_embedding_dim
        self.text_dropout = nn.Dropout(text_dropout_rate)
        self.audio_dropout = nn.Dropout(audio_dropout_rate)
        self.dropout = nn.Dropout(max(text_dropout_rate, audio_dropout_rate))

        if fusion == 'early':
            self.joint = nn.Sequential(nn.Linear(d_text + d_audio, d_model), nn.ReLU())
            self.head = classification_head(d_model, num_classes)

        elif fusion in ('intermediate', 'late'):
            self.text_branch = UnimodalEncoder(d_text, d_model, n_heads, text_dropout_rate)
            self.audio_branch = UnimodalEncoder(d_audio, d_model, n_heads, audio_dropout_rate)
            if fusion == 'intermediate':
                self.joint = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())
                self.head = classification_head(d_model, num_classes)
            else:
                self.text_head = classification_head(d_model, num_classes)
                self.audio_head = classification_head(d_model, num_classes)
                # Logits of the two classifiers mixed with softmax(modality_logits); starts at 50/50.
                self.modality_logits = nn.Parameter(th.zeros(2))

        elif fusion == 'textonly':
            self.text_branch = UnimodalEncoder(d_text, d_model, n_heads, text_dropout_rate)
            self.text_head = classification_head(d_model, num_classes)

        elif fusion == 'selfattn':
            self.text_projection = nn.Linear(d_text, d_model)
            self.audio_projection = nn.Linear(d_audio, d_model)
            self.modality_embedding = nn.Embedding(2, d_model)  # 0 = text, 1 = audio
            self.joint_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                          dim_feedforward=2 * d_model,
                                                          dropout=text_dropout_rate, batch_first=True)
            self.joint = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
            self.head = classification_head(d_model, num_classes)

        elif fusion == 'crossattn':
            self.text_projection = nn.Linear(d_text, d_model)
            self.audio_projection = nn.Linear(d_audio, d_model)
            self.text_from_audio = CrossAttentionBlock(d_model, n_heads, text_dropout_rate)
            self.audio_from_text = CrossAttentionBlock(d_model, n_heads, audio_dropout_rate)
            self.joint = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())
            self.head = classification_head(d_model, num_classes)

    def modality_weights(self):
        """`late` only: current (text, audio) weights of the decision-level mixture."""
        return th.softmax(self.modality_logits, dim=0).detach().cpu()

    def forward(self, inputs):
        text_mask = inputs['text_input_mask']
        audio, audio_mask = inputs['audio_inputs'], inputs['audio_input_mask']

        # [bs, T, d_text] / [bs, A, d_audio]
        tokens = self.text_dropout(self.text_encoder(inputs['text_inputs'], text_mask))
        frames = self.audio_dropout(audio)

        if self.fusion == 'textonly':
            return self.text_head(self.text_branch(tokens, text_mask))

        if self.fusion == 'early':
            fused = th.cat((masked_mean(tokens, text_mask), masked_mean(frames, audio_mask)), dim=-1)
            return self.head(self.dropout(self.joint(fused)))

        if self.fusion in ('intermediate', 'late'):
            text_emb = self.text_branch(tokens, text_mask)
            audio_emb = self.audio_branch(frames, audio_mask)
            if self.fusion == 'intermediate':
                fused = self.joint(th.cat((text_emb, audio_emb), dim=-1))
                return self.head(self.dropout(fused))
            weights = th.softmax(self.modality_logits, dim=0)
            return weights[0] * self.text_head(text_emb) + weights[1] * self.audio_head(audio_emb)

        text_seq = self.text_projection(tokens)
        audio_seq = self.audio_projection(frames)

        if self.fusion == 'selfattn':
            text_seq = text_seq + self.modality_embedding.weight[0]
            audio_seq = audio_seq + self.modality_embedding.weight[1]
            sequence = th.cat((text_seq, audio_seq), dim=1)
            mask = th.cat((text_mask.to(audio_mask.dtype), audio_mask), dim=1)
            hidden = self.joint_layer(sequence, src_key_padding_mask=padding_mask(mask))
            return self.head(self.dropout(self.joint(masked_mean(hidden, mask))))

        # crossattn
        text_hidden = self.text_from_audio(text_seq, audio_seq, audio_mask)
        audio_hidden = self.audio_from_text(audio_seq, text_seq, text_mask)
        fused = th.cat((masked_mean(text_hidden, text_mask), masked_mean(audio_hidden, audio_mask)), dim=-1)
        return self.head(self.dropout(self.joint(fused)))
