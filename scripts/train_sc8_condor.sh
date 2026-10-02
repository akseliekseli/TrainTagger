#!/bin/bash
set -euo pipefail

REPOSITORY="/eos/home-a/asuutari/projects/TrainTagger"
PYTHON="/eos/home-a/asuutari/conda-envs/tagger/bin/python"
DATA_DIR="/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd-minbias-label-selected"

MODEL_NAME="test_stream"
MODEL_CONFIG="MLPmixer_HGQ2.yaml"
CLASSES="sc8_classes.yaml"
PERCENT=100
EBOPS=300000

MODEL_DIR="/eos/home-a/asuutari/projects/TrainTagger/output/${MODEL_NAME}"
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
echo "Dataset percentage: ${PERCENT}"

"${PYTHON}" -u -m tagger.train.train \
    --yaml_config tagger/model/configs/"${MODEL_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --test-data-dirs \
    /eos/home-a/asuutari/FastPUPPI/XtoHH-qcd-minbias/signal_process_data/XtoHH \
    /eos/home-a/asuutari/FastPUPPI/XtoHH-qcd-minbias/signal_process_data/MinBias \
    --class-config "${CLASS_CONFIG}" \
    --output "${MODEL_DIR}" \
    --percent "${PERCENT}" \
    --ebops "${EBOPS}" \
    --stream-read-entries 8192 \
    --stream-shuffle-jets 10000 \
    --stream-prefetch 1

echo "Starting plotting"

"${PYTHON}" -u -m tagger.train.train \
    --plot-basic \
    --output "${MODEL_DIR}"

echo "Plotting completed at $(date)"
echo "Results saved in: ${MODEL_DIR}"
