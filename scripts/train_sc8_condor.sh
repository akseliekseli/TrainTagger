#!/bin/bash
set -euo pipefail

REPOSITORY="/eos/home-a/asuutari/projects/TrainTagger"
PYTHON="/eos/home-a/asuutari/conda-envs/tagger/bin/python"
DATA_DIR="/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd-v2"
CLASS_CONFIG="${REPOSITORY}/tagger/train/sc8_classes.yaml"

PERCENT=100
MODEL_DIR="/eos/home-a/asuutari/projects/TrainTagger/output/baseline_sc8_mlpmix_HGQ2"

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

nvidia-smi

"${PYTHON}" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)

if not gpus:
    raise RuntimeError("No GPU is visible to TensorFlow")
PY

"${PYTHON}" -u -m tagger.train.train \
    --yaml_config tagger/model/configs/MLPmixer_HGQ2.yaml \
    --data-dir "${DATA_DIR}" \
    --class-config "${CLASS_CONFIG}" \
    --output "${MODEL_DIR}" \
    --percent "${PERCENT}" \
    --ebops 300000

echo "Starting plotting"

"${PYTHON}" -u -m tagger.train.train \
    --plot-basic \
    --output "${MODEL_DIR}"

echo "Plotting completed at $(date)"
echo "Results saved in: ${MODEL_DIR}"
