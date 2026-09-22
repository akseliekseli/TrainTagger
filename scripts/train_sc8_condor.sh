#!/bin/bash
set -euo pipefail

REPOSITORY="/eos/home-a/asuutari/projects/TrainTagger"
PYTHON="/eos/home-a/asuutari/conda-envs/tagger/bin/python"

cd "${REPOSITORY}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${_CONDOR_SCRATCH_DIR}/matplotlib"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

nvidia-smi

"${PYTHON}" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)

if not gpus:
    raise RuntimeError("No GPU is visible to TensorFlow")
PY

exec "${PYTHON}" -m tagger.train.train \
    --yaml_config tagger/model/configs/baseline_sc8_HGQ2.yaml \
    --data-dir /eos/home-a/asuutari/FastPUPPI/XtoHH-qcd \
    --output /eos/home-a/asuutari/projects/TrainTagger/output/baseline_sc8_HGQ2_full \
    --percent 100 \
    --ebops 300000
