#!/usr/bin/env bash
# Data exploration: E1 labels and text, E2 audio, E3 alignment audit (Whisper, GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m data_exploration.e1_text_labels
python -m data_exploration.e2_audio
python -m data_exploration.e3_alignment_whisper "$@"
