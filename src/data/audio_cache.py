"""On-disk cache of frozen audio-encoder features (drop-in replacement of MAMKit's AudioTransformer)."""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
from tqdm import tqdm

from mamkit.data.processing import AudioTransformer
from mamkit.utility.processing import encode_audio_nn

from src.paths import FEATURE_CACHE_DIR

logger = logging.getLogger(__name__)


def _clip_id(path) -> str:
    path = Path(path)
    return f'{path.parent.name}/{path.name}'


def _is_cacheable(audio_input) -> bool:
    # Same early-exit cases as encode_audio_nn (placeholder tensor, no encoder involved).
    if audio_input is None:
        return False
    if isinstance(audio_input, Path):
        return audio_input.exists()
    return len(audio_input) > 0


class CachedAudioTransformer(AudioTransformer):

    def __init__(self, *args, cache_dir: Path = FEATURE_CACHE_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        config = {
            'model_card': self.model_card,
            'sampling_rate': self.sampling_rate,
            'downsampling_factor': self.downsampling_factor,
            'aggregate': self.aggregate,
            'processor_args': self.processor_args,
            'model_args': self.model_args,
        }
        config_hash = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
        model_name = self.model_card.replace('/', '--')
        self.cache_dir = Path(cache_dir) / 'audio' / f'{model_name}__{config_hash}'
        self.n_hits = self.n_misses = 0

    def _cache_path(self, audio_input) -> Path:
        clips = [audio_input] if isinstance(audio_input, Path) else list(audio_input)
        key = hashlib.sha1('|'.join(_clip_id(c) for c in clips).encode()).hexdigest()
        return self.cache_dir / key[:2] / f'{key}.npy'

    def _encode(self, audio_input):
        if not _is_cacheable(audio_input):
            return encode_audio_nn(audio_input=audio_input, processor=self.processor, model=self.model,
                                   device=self.device)

        path = self._cache_path(audio_input)
        if path.exists():
            self.n_hits += 1
            return np.load(path)

        self.n_misses += 1
        if self.model is None:
            self._init_models()
        features = encode_audio_nn(audio_input=audio_input,
                                   model=self.model,
                                   processor=self.processor,
                                   processor_args=self.processor_args,
                                   model_args=self.model_args,
                                   device=self.device,
                                   sampling_rate=self.sampling_rate,
                                   downsampling_factor=self.downsampling_factor,
                                   aggregate=self.aggregate)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')
        np.save(tmp_path, features)
        os.replace(tmp_path, path)
        return features

    def __call__(
            self,
            inputs: List[Union[Path, List[Path]]],
            context: List[List[Path]] = None
    ):
        input_context = context if context is not None else [None] * len(inputs)

        input_features, context_features = [], []
        for audio_input, audio_context in tqdm(zip(inputs, input_context),
                                               desc='Loading/extracting audio features (cached)...',
                                               total=len(inputs)):
            input_features.append(self._encode(audio_input))
            if audio_context is not None:
                context_features.append(self._encode(audio_context))

        logger.info(f'Feature cache {self.cache_dir.name}: {self.n_hits} loaded, {self.n_misses} computed')
        self.n_hits = self.n_misses = 0
        return input_features, context_features if len(context_features) else None
