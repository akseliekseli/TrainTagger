#!/bin/bash
set -euo pipefail

REPOSITORY="/eos/home-a/asuutari/projects/TrainTagger"
PYTHON="/eos/home-a/asuutari/conda-envs/tagger/bin/python"
DATA_DIR="/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd-minbias/"

MODEL_NAME="ds_x_multiclass"
MODEL_CONFIG="baseline_HGQ2.yaml"
CLASSES="multiclass_classes.yaml"
PERCENT=100
EBOPS=300000

MODEL_DIR="${REPOSITORY}/output/${MODEL_NAME}"
CLASS_CONFIG="${REPOSITORY}/tagger/train/configs/${CLASSES}"

cd "${REPOSITORY}"

export PYTHONUNBUFFERED=1

SCRATCH_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="${SCRATCH_DIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "$(dirname "${MODEL_DIR}")"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Model directory: ${MODEL_DIR}"
echo "Dataset: ${DATA_DIR}"
echo "Dataset percentage: ${PERCENT}"

"${PYTHON}" -u -m tagger.train.train \
    --yaml_config "${REPOSITORY}/tagger/model/configs/${MODEL_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --test-data-dirs \
    "${DATA_DIR}/signal_process_data/XtoHH" \
    "${DATA_DIR}/signal_process_data/MinBias" \
    --class-config "${CLASS_CONFIG}" \
    --output "${MODEL_DIR}" \
    --percent "${PERCENT}" \
    --ebops "${EBOPS}"

echo "Starting plotting"

"${PYTHON}" -u -m tagger.train.train \
    --plot-basic \
    --data-dir "${DATA_DIR}" \
    --output "${MODEL_DIR}"

echo "Plotting completed at $(date)"
echo "Results saved in: ${MODEL_DIR}"
