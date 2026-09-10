#!/usr/bin/env bash
# Create the isolated `rlsdc` conda environment on the workstation (dtgpu).
#
# Safe to re-run: skips environment creation if it already exists.
#
# The PyTorch index is pinned to cu128 deliberately. The RTX 5060 Ti is
# Blackwell (compute capability 12.0 / sm_120), and cu121/cu124 wheels contain
# no sm_120 kernels -- they install cleanly and then fail at the first GPU op.
# See docs/01_infrastructure_and_workflow.md.

set -euo pipefail

ENV_NAME="rlsdc"
PY_VERSION="3.11"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

# conda is not on PATH for non-interactive ssh, so source it explicitly.
source ~/miniforge3/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] env '${ENV_NAME}' already exists, skipping creation"
else
    echo "[setup] creating env '${ENV_NAME}' (python ${PY_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"
echo "[setup] active env: ${CONDA_DEFAULT_ENV}"
echo "[setup] python: $(python --version), at $(which python)"

# Hard guard. This machine is shared and other projects have their own torch
# builds (env_isaaclab, lerobot). Abort rather than risk installing into the
# wrong environment, base, or user site-packages.
EXPECTED_PY="${HOME}/miniforge3/envs/${ENV_NAME}/bin/python"
ACTUAL_PY="$(which python)"
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" || "${ACTUAL_PY}" != "${EXPECTED_PY}" ]]; then
    echo "[setup] ABORT: wrong environment active." >&2
    echo "        expected env '${ENV_NAME}' at ${EXPECTED_PY}" >&2
    echo "        got      env '${CONDA_DEFAULT_ENV}' at ${ACTUAL_PY}" >&2
    exit 1
fi
echo "[setup] isolation guard passed"

# Never --user, never sudo: both would escape the environment.
export PIP_USER=0

echo "[setup] installing pytorch from ${TORCH_INDEX}"
pip install --upgrade pip
pip install torch --index-url "${TORCH_INDEX}"

echo "[setup] installing project dependencies"
pip install \
    "gymnasium" \
    "highway-env" \
    "stable-baselines3" \
    "tensorboard" \
    "pytest" \
    "pyyaml" \
    "numpy" \
    "matplotlib" \
    "imageio" \
    "imageio-ffmpeg"

echo "[setup] done. verify with: bash scripts/verify_env.sh"
