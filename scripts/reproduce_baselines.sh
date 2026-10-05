#!/usr/bin/env bash
# Baseline Transformer, 2 tasks x 3 modalities, 3 seeds (Table 4 of the shared-task paper).
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="42 2024 666"

python -m src.data.prepare

for task in afc afd; do
  for modality in text audio text_audio; do
    python -m src.experiments.baseline --task "$task" --modality "$modality" --seeds $SEEDS
  done
done

python -m src.evaluation.aggregate
python -m src.evaluation.per_class
