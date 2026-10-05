"""Training/evaluation loop over seeds, official MM-ArgFallacy2025 split and validation debates."""
import logging
from pathlib import Path
from typing import Callable, Sequence

import lightning as L
import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import F1Score

from mamkit.utility.callbacks import PycharmProgressBar
from mamkit.utility.model import MAMKitLightingModel

from src.training.imbalance import build_imbalance_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_f1_metric(num_classes: int) -> F1Score:
    """Binary F1 for AFD, macro F1 for AFC."""
    if num_classes == 2:
        return F1Score(task='binary')
    return F1Score(task='multiclass', num_classes=num_classes, average='macro')


def save_test_predictions(trainer, test_dataloader, save_path: Path, seed: int, fold: int):
    """Save test labels and predictions of the best checkpoint."""
    y_pred = np.concatenate(trainer.predict(ckpt_path='best', dataloaders=test_dataloader))
    y_true = np.concatenate([batch[1].cpu().numpy() for batch in test_dataloader])
    np.savez(save_path.joinpath(f'test_predictions_seed{seed}_fold{fold}.npz').as_posix(), y_true=y_true, y_pred=y_pred)


SHARED_TASK_SPLIT = 'mm-argfallacy-2025'
SHARED_TASK_TEST_DIALOGUES = ('47_2024', '48_2024')
N_VAL_DIALOGUES = 4


def split_shared_task_dialogues(dialogue_ids, n_val_dialogues: int = N_VAL_DIALOGUES):
    """(train, val, test) debate ids of the official split + our validation hold-out."""
    test = sorted(set(dialogue_ids) & set(SHARED_TASK_TEST_DIALOGUES))
    candidates = sorted(set(dialogue_ids) - set(SHARED_TASK_TEST_DIALOGUES))
    val = sorted(str(d) for d in np.random.choice(candidates, size=n_val_dialogues, replace=False))
    train = [d for d in candidates if d not in val]
    return train, val, test


def shared_task_split(loader, n_val_dialogues: int = N_VAL_DIALOGUES):
    """Official split with a validation set of whole training debates."""
    data = loader.data
    if loader.task_name == 'afc':
        data = data[data['fallacy'].notna()]
    _, val_dialogues, test_dialogues = split_shared_task_dialogues(data['dialogue_id'].values, n_val_dialogues)
    is_test = data['dialogue_id'].isin(test_dialogues)
    train_df, test_df = data[~is_test], data[is_test]
    is_val = train_df['dialogue_id'].isin(val_dialogues)
    logger.info(f'Validation debates: {val_dialogues}')

    return loader.build_info_from_splits(train_df=train_df[~is_val],
                                         val_df=train_df[is_val],
                                         test_df=test_df)


def get_splits(loader, split_key: str):
    if split_key == SHARED_TASK_SPLIT:
        return [shared_task_split(loader)]
    return loader.get_splits(key=split_key)


def run_training_loop(
        loader,
        split_key: str,
        seeds: Sequence[int],
        build_processor: Callable,
        build_collator: Callable,
        build_model: Callable,
        num_classes: int,
        loss_function,
        optimizer_class,
        optimizer_args: dict,
        batch_size: int,
        trainer_args: dict,
        save_path: Path,
        imbalance: str = 'weighted',
        save_model: bool = False,
):
    """Train and evaluate one configuration over seeds; save metrics and test predictions."""
    save_path.mkdir(parents=True, exist_ok=True)
    metrics = {}

    for seed in seeds:
        seed_everything(seed=seed)
        for fold, split_info in enumerate(get_splits(loader, split_key)):
            processor = build_processor()
            processor.fit(split_info.train)

            split_info.train = processor(split_info.train)
            split_info.val = processor(split_info.val)
            split_info.test = processor(split_info.test)
            processor.clear()

            # Re-seed after preprocessing: loading a frozen encoder consumes RNG draws.
            seed_everything(seed=seed + fold)

            collator = build_collator()

            train_labels = np.asarray(split_info.train.labels, dtype=int)
            fold_loss_function, train_sampler, imbalance_info = build_imbalance_strategy(
                imbalance, loss_function, train_labels, num_classes)
            metrics.setdefault('imbalance', []).append(imbalance_info)
            logger.info(f'[seed={seed}] imbalance: {imbalance_info}')

            train_dataloader = DataLoader(split_info.train, batch_size=batch_size,
                                          shuffle=train_sampler is None, sampler=train_sampler,
                                          collate_fn=collator)
            val_dataloader = DataLoader(split_info.val, batch_size=batch_size,
                                        shuffle=False, collate_fn=collator)
            test_dataloader = DataLoader(split_info.test, batch_size=batch_size,
                                         shuffle=False, collate_fn=collator)

            model = MAMKitLightingModel(
                model=build_model(),
                loss_function=fold_loss_function,
                num_classes=num_classes,
                optimizer_class=optimizer_class,
                val_metrics=MetricCollection({'f1': build_f1_metric(num_classes)}),
                test_metrics=MetricCollection({'f1': build_f1_metric(num_classes)}),
                **optimizer_args,
            )

            trainer = L.Trainer(
                **trainer_args,
                callbacks=[
                    EarlyStopping(monitor='val_loss', mode='min', patience=5),
                    ModelCheckpoint(monitor='val_loss', mode='min'),
                    PycharmProgressBar(),
                ],
            )
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            val_metrics = trainer.test(ckpt_path='best', dataloaders=val_dataloader)[0]
            test_metrics = trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]
            save_test_predictions(trainer, test_dataloader, save_path, seed, fold)
            if save_model:
                th.save(model.model.state_dict(), save_path / f'model_seed{seed}_fold{fold}.pt')
            logger.info(f'[seed={seed}] validation metrics: {val_metrics}')
            logger.info(f'[seed={seed}] test metrics: {test_metrics}')

            for metric_name, metric_value in val_metrics.items():
                metrics.setdefault('validation', {}).setdefault(metric_name, []).append(metric_value)
            for metric_name, metric_value in test_metrics.items():
                metrics.setdefault('test', {}).setdefault(metric_name, []).append(metric_value)

            processor.reset()

    metric_names = list(metrics['validation'].keys())
    for split_name in ['validation', 'test']:
        for metric_name in metric_names:
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(seeds), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (per_seed_avg.mean(), per_seed_avg.std())

    logger.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics, allow_pickle=True)
    return metrics
