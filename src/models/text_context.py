"""Dialogue context for the text model: RoBERTa sentence pair (context, target), optional target-only pooling."""
import torch as th
from transformers import AutoTokenizer

from mamkit.models.text import Transformer

CONTEXT_POOLINGS = ('all', 'target')


class ContextTextCollator:
    """Tokenizes (context, target) pairs; optionally returns a mask of the target tokens."""

    def __init__(self, model_card: str, max_length: int = 512, target_pooling: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.max_length = max_length
        self.target_pooling = target_pooling

    def __call__(self, inputs, context=None):
        if context is None:
            context = [''] * len(inputs)
        encoded = self.tokenizer([c if c else '' for c in context], [str(t) for t in inputs],
                                 padding=True, truncation='only_first', max_length=self.max_length,
                                 return_tensors='pt')
        out = {'inputs': encoded['input_ids'], 'input_mask': encoded['attention_mask']}
        if self.target_pooling:
            out['pool_mask'] = th.tensor([[1 if s == 1 else 0 for s in encoded.sequence_ids(i)]
                                          for i in range(len(inputs))], dtype=encoded['attention_mask'].dtype)
        return out


class TargetPooledTransformer(Transformer):
    """MAMKit text Transformer whose mean pooling runs over the target tokens only."""

    def forward(self, inputs):
        attention_mask = inputs['input_mask']
        pool_mask = inputs.get('pool_mask', attention_mask)
        tokens_emb = self.model(input_ids=inputs['inputs'], attention_mask=attention_mask).last_hidden_state
        tokens_emb = self.dropout(tokens_emb)
        text_emb = (tokens_emb * pool_mask[:, :, None]).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)[:, None]
        return self.head(text_emb)
