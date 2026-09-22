#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/eos/home-a/asuutari/projects/TrainTagger"
PYTHON="/eos/home-a/asuutari/conda-envs/tagger/bin/python"
DATA_DIR="/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd"
MODEL_DIR="/eos/home-a/asuutari/FastPUPPI/models/baseline_sc8_HGQ2_full_retry"
CONFIG="${PROJECT_DIR}/tagger/model/configs/baseline_sc8_HGQ2.yaml"

cd "${PROJECT_DIR}"

echo "Starting training at $(date)"

"${PYTHON}" -u -m tagger.train.train \
    --yaml_config "${CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --output "${MODEL_DIR}" \
    --percent 100 \
    --ebops 300000

echo "Training completed successfully at $(date)"
echo "Starting basic plots"

"${PYTHON}" -u -m tagger.train.train \
    --plot-basic \
    --output "${MODEL_DIR}"

echo "Basic plots completed successfully at $(date)"
