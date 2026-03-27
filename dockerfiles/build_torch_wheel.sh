#!/bin/bash
set -euo pipefail
set -x

# Build a CPU-only, C++11 ABI enabled PyTorch wheel and place it under ${OUT_DIR}.
# Defaults can be overridden via environment variables.
TORCH_VERSION=${TORCH_VERSION:-2.7.1}
TORCH_BRANCH=${TORCH_BRANCH:-v${TORCH_VERSION}}
PYTHON_BIN=${PYTHON_BIN:-python3}
OUT_DIR=${OUT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/binary}
SRC_DIR=${SRC_DIR:-${OUT_DIR}/pytorch}

# CXX11 ABI tag for wheel naming
CXX11_ABI_TAG=cxx11abi

# Parallelism (conservative for CI)
export MAX_JOBS=${MAX_JOBS:-2}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

# Skip if wheel already exists
if ls "${OUT_DIR}/torch-${TORCH_VERSION}"*cxx11*linux_x86_64.whl >/dev/null 2>&1; then
  echo "Found existing torch-${TORCH_VERSION} cxx11 wheel under ${OUT_DIR}, skip build."
  exit 0
fi

# Try to download prebuilt CPU + CXX11 ABI wheel first
PY_TAG=$(${PYTHON_BIN} - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)
PREBUILT_NAME="torch-${TORCH_VERSION}+cpu.cxx11.abi-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
PREBUILT_NAME_ENC=$(printf "%s" "${PREBUILT_NAME}" | sed 's/+/%2B/g')
PREBUILT_URL="https://download.pytorch.org/whl/cpu-cxx11-abi/${PREBUILT_NAME_ENC}"

mkdir -p "${OUT_DIR}"
echo "Attempting to download prebuilt wheel: ${PREBUILT_URL}"
if curl -fL --retry 3 --retry-delay 2 -o "${OUT_DIR}/${PREBUILT_NAME}" "${PREBUILT_URL}"; then
  echo "Downloaded prebuilt ${PREBUILT_NAME} to ${OUT_DIR}"
  exit 0
else
  echo "Prebuilt wheel not available, falling back to source build"
fi

# Prereqs for building PyTorch CPU wheel
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  cmake ninja-build \
  python3-dev python3-pip python3-setuptools python3-wheel \
  libopenblas-dev libblas-dev liblapack-dev libatlas-base-dev

# Clone or update source
mkdir -p "${OUT_DIR}"
if [ ! -d "${SRC_DIR}" ]; then
  git clone --branch "${TORCH_BRANCH}" --recursive https://github.com/pytorch/pytorch.git "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --tags --force
  git -C "${SRC_DIR}" checkout "${TORCH_BRANCH}"
  git -C "${SRC_DIR}" submodule sync --recursive
  git -C "${SRC_DIR}" submodule update --init --recursive
fi

cd "${SRC_DIR}"

# CPU-only, C++11 ABI build knobs
export USE_CUDA=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=0
export USE_ONNX=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_NNPACK=0
export BUILD_TEST=0
export USE_KINETO=0
export USE_FBGEMM=0
export USE_CXX11_ABI=1

export PIP_NO_BUILD_ISOLATION=1
export PIP_PROGRESS_BAR=off

export CMAKE_POLICY_VERSION_MINIMUM=3.5

${PYTHON_BIN} -m pip install --no-cache-dir -U pip setuptools wheel typing_extensions numpy pyyaml ninja
if [ -f requirements.txt ]; then
  ${PYTHON_BIN} -m pip install --no-cache-dir -r requirements.txt || true
fi

${PYTHON_BIN} -m pip wheel . --no-build-isolation -w "${OUT_DIR}"

echo "Built torch ${TORCH_VERSION} ${CXX11_ABI_TAG} wheel(s) under ${OUT_DIR}:"
ls -lh ${OUT_DIR}/torch-${TORCH_VERSION}*-linux_x86_64.whl
