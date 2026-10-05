"""Text, audio and text-audio runners built on the MAMKit Baseline Transformer configurations."""
import torch as th
from mamkit.configs.audio import TransformerEncoderConfig
from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.configs.text_audio import MMTransformerConfig
from mamkit.data.collators import (
    AudioCollatorOutput,
    MultimodalCollator,
    TextTransformerCollator,
    UnimodalCollator,
)
from mamkit.data.datasets import InputMode, MMUSEDFallacy
from mamkit.data.processing import MultimodalProcessor, UnimodalProcessor
from mamkit.models.audio import TransformerEncoder
from mamkit.models.text import Transformer
from mamkit.models.text_audio import MMTransformer

from src.data.audio_cache import CachedAudioTransformer
from src.models.fusion import TextAudioFusionModel
from src.models.text_context import ContextTextCollator, TargetPooledTransformer
from src.paths import BASE_DATA_PATH, RESULTS_DIR
from src.training.loop import SHARED_TASK_SPLIT, run_training_loop
from src.training.options import BASELINE, ExperimentOptions

SPLIT_KEYS = (SHARED_TASK_SPLIT, 'mancini-et-al-2024')

_TRAINER_DEFAULTS = {'accelerator': 'auto', 'devices': 1, 'accumulate_grad_batches': 3}


def load_config(config_cls, key: ConfigKey):
    """Equality-based replacement for mamkit's `config_cls.from_config(key)`."""
    for stored_key, method_name in config_cls.configs.items():
        if stored_key == key:
            return getattr(config_cls, method_name)()
    raise KeyError(key)


def _valid_task(task_name: str):
    if task_name not in ('afc', 'afd'):
        raise ValueError(f"task_name must be 'afc' or 'afd', got {task_name!r}")


def result_dir(split_key: str, task_name: str, run_name: str):
    return RESULTS_DIR / 'mmused-fallacy' / split_key / task_name / run_name


def _audio_processor(config, model_card: str, model_args):
    return CachedAudioTransformer(
        model_card=model_card,
        processor_args=config.processor_args,
        model_args=model_args,
        aggregate=config.aggregate,
        downsampling_factor=config.downsampling_factor,
        sampling_rate=config.sampling_rate,
    )


def _train(loader, config, options: ExperimentOptions, split_key, task_name, base_run_name, seeds,
           max_epochs, build_processor, build_collator, build_model, save_model=False):
    """Shared tail of the three runners: options -> optimizer / imbalance / results directory."""
    return run_training_loop(
        loader=loader, split_key=split_key, seeds=seeds or config.seeds,
        build_processor=build_processor, build_collator=build_collator, build_model=build_model,
        num_classes=config.num_classes, loss_function=config.loss_function,
        optimizer_class=config.optimizer, optimizer_args=options.optimizer_args(config.optimizer_args),
        batch_size=config.batch_size, trainer_args={**_TRAINER_DEFAULTS, 'max_epochs': max_epochs},
        save_path=result_dir(split_key, task_name, options.run_name(base_run_name)),
        imbalance=options.imbalance, save_model=save_model,
    )


def run_text_only(task_name: str, seeds=None, max_epochs: int = 50, split_key: str = SHARED_TASK_SPLIT,
                  options: ExperimentOptions = BASELINE, save_model: bool = False):
    _valid_task(task_name)
    if options.fusion != 'concat':
        raise ValueError('fusion options only apply to the text-audio modality')
    loader_args = {'with_context': True, 'context_window': options.context} if options.context else {}
    loader = MMUSEDFallacy(task_name=task_name, input_mode=InputMode.TEXT_ONLY, base_data_path=BASE_DATA_PATH,
                           **loader_args)
    config = load_config(TransformerConfig, ConfigKey(
        dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name=task_name,
        tags={'anonymous', 'roberta'},
    ))

    def build_processor():
        return UnimodalProcessor()

    def build_collator():
        if options.context:
            text_collator = ContextTextCollator(model_card=config.model_card,
                                                target_pooling=options.context_pooling == 'target')
        else:
            text_collator = TextTransformerCollator(model_card=config.model_card, tokenizer_args=config.tokenizer_args)
        return UnimodalCollator(features_collator=text_collator, label_collator=lambda labels: th.tensor(labels))

    def build_model():
        model_cls = TargetPooledTransformer if options.context and options.context_pooling == 'target' else Transformer
        return model_cls(
            model_card=config.model_card,
            is_transformer_trainable=options.text_encoder_trainable(config.is_transformer_trainable),
            dropout_rate=config.dropout_rate,
            head=config.head,
        )

    return _train(loader, config, options, split_key, task_name, 'text_only_roberta', seeds, max_epochs,
                  build_processor, build_collator, build_model, save_model)


def run_audio_only(task_name: str, seeds=None, max_epochs: int = 20, split_key: str = SHARED_TASK_SPLIT,
                   options: ExperimentOptions = BASELINE, save_model: bool = False):
    _valid_task(task_name)
    if options.fusion != 'concat' or options.finetune or options.context:
        raise ValueError('fusion / finetune / context options need a text encoder: not available for audio-only')
    loader = MMUSEDFallacy(task_name=task_name, input_mode=InputMode.AUDIO_ONLY, base_data_path=BASE_DATA_PATH)
    config = load_config(TransformerEncoderConfig, ConfigKey(
        dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name=task_name,
        tags={'anonymous', 'wavlm'},
    ))

    def build_processor():
        return UnimodalProcessor(features_processor=_audio_processor(config, config.model_card, config.model_args))

    def build_collator():
        return UnimodalCollator(
            features_collator=AudioCollatorOutput(),
            label_collator=lambda labels: th.tensor(labels),
        )

    def build_model():
        return TransformerEncoder(
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate,
            encoder=config.encoder,
            head=config.head,
        )

    return _train(loader, config, options, split_key, task_name, 'audio_only_transformer_wavlm', seeds,
                  max_epochs, build_processor, build_collator, build_model, save_model)


def run_text_audio(task_name: str, seeds=None, max_epochs: int = 20, split_key: str = SHARED_TASK_SPLIT,
                   options: ExperimentOptions = BASELINE, save_model: bool = False):
    """options.fusion='concat' = Table 4 baseline (mamkit MMTransformer, unchanged)."""
    _valid_task(task_name)
    if options.context:
        raise ValueError('context is implemented for the text modality only')
    loader = MMUSEDFallacy(task_name=task_name, input_mode=InputMode.TEXT_AUDIO, base_data_path=BASE_DATA_PATH)
    config = load_config(MMTransformerConfig, ConfigKey(
        dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name=task_name,
        tags={'anonymous', 'roberta', 'wavlm'},
    ))
    text_trainable = options.text_encoder_trainable(config.is_transformer_trainable)

    def build_processor():
        return MultimodalProcessor(
            audio_processor=_audio_processor(config, config.audio_model_card, config.audio_model_args))

    def build_collator():
        return MultimodalCollator(
            text_collator=TextTransformerCollator(model_card=config.text_model_card,
                                                   tokenizer_args=config.tokenizer_args),
            audio_collator=AudioCollatorOutput(),
            label_collator=lambda labels: th.tensor(labels),
        )

    def build_model():
        if options.fusion != 'concat':
            return TextAudioFusionModel(
                fusion=options.fusion,
                model_card=config.text_model_card,
                num_classes=config.num_classes,
                audio_embedding_dim=config.audio_embedding_dim,
                text_dropout_rate=config.text_dropout_rate,
                audio_dropout_rate=config.audio_dropout_rate,
                is_transformer_trainable=text_trainable,
            )
        return MMTransformer(
            model_card=config.text_model_card,
            head=config.head,
            text_dropout_rate=config.text_dropout_rate,
            audio_dropout_rate=config.audio_dropout_rate,
            is_transformer_trainable=text_trainable,
            lstm_weights=config.lstm_weights,
            audio_embedding_dim=config.audio_embedding_dim,
        )

    return _train(loader, config, options, split_key, task_name, 'text_audio_roberta_wavlm', seeds, max_epochs,
                  build_processor, build_collator, build_model, save_model)
