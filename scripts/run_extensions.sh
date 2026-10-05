#!/usr/bin/env bash
# All extensions of the report, 3 seeds each. Requires scripts/reproduce_baselines.sh to have run.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="42 2024 666"
TEACHER=text_only_roberta_ft_imb-weighted_train

for task in afc afd; do
  # Fine-tuning and class-imbalance strategies
  python -m src.experiments.baseline --task $task --modality text --finetune --seeds $SEEDS
  for imb in weighted_train focal sampler; do
    save=""; [ "$task-$imb" = "afc-weighted_train" ] && save="--save-model"  # AFC teacher for cascade/distill/XAI
    python -m src.experiments.baseline --task $task --modality text --finetune --imbalance $imb $save --seeds $SEEDS
  done
  python -m src.experiments.baseline --task $task --modality text_audio --finetune --seeds $SEEDS

  # Dialogue context: mean over the pair, then over the target tokens only
  for k in 1 2; do
    python -m src.experiments.baseline --task $task --modality text --context $k --seeds $SEEDS
    python -m src.experiments.baseline --task $task --modality text --context $k --context-pooling target --seeds $SEEDS
  done

  # Multi-task learning (lambda = 0.5) and its lambda = 0 control
  for aux in asd none; do
    python -m src.experiments.multitask --task $task --aux $aux --seeds $SEEDS
  done
done

# Context with the fine-tuned encoder
python -m src.experiments.baseline --task afc --modality text --finetune --imbalance weighted_train \
  --context 1 --seeds $SEEDS

# Fusion mechanisms and learning-rate selection on the validation set
for fusion in early intermediate late selfattn crossattn textonly; do
  python -m src.experiments.baseline --task afc --modality text_audio --fusion $fusion --seeds $SEEDS
  for lr in 1e-4 5e-5; do
    python -m src.experiments.baseline --task afc --modality text_audio --fusion $fusion --lr $lr --seeds $SEEDS
  done
done
python -m src.evaluation.select_lr --task afc --lrs 1e-4 5e-5

# Audio only: Whisper cascade (needs data_exploration/run_all.sh for the transcripts)
python -m src.experiments.cascade --text-run $TEACHER --seeds $SEEDS

# Text-to-audio distillation and its controls
python -m src.experiments.distill --mode raw --seeds $SEEDS
python -m src.experiments.distill --mode supervised --seeds $SEEDS
for teacher in finetuned generic; do
  python -m src.experiments.distill --teacher $teacher --teacher-temp 0.02 --label-fractions 0.1 0.25 0.5 --seeds $SEEDS
done

# Explainability of the AFC text model
python -m src.experiments.xai --text-run $TEACHER --seeds $SEEDS

# Macro-F1 conventions and paired bootstrap table
python -m src.evaluation.posthoc
