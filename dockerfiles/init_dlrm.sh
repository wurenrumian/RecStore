#!/bin/bash
cd "$(dirname "$0")"
set -euo pipefail
set -x

PYTHON_BIN="${PYTHON_BIN:-python3}"
MIRROR=1
WORKSPACE_ROOT="${DLRM_BUILD_ROOT:-/tmp/recstore-dlrm-deps}"
FBGEMM_REPO="${FBGEMM_REPO:-https://github.com/pytorch/FBGEMM.git}"
FBGEMM_REF="${FBGEMM_REF:-v1.1.2}"
TORCHREC_REPO="${TORCHREC_REPO:-https://github.com/pytorch/torchrec.git}"
TORCHREC_REF="${TORCHREC_REF:-v1.1.0}"
PINNED_TORCHREC_VERSION="${PINNED_TORCHREC_VERSION:-1.2.0}"
PINNED_FBGEMM_GPU_VERSION="${PINNED_FBGEMM_GPU_VERSION:-1.2.0}"
PINNED_TORCHMETRICS_VERSION="${PINNED_TORCHMETRICS_VERSION:-1.0.3}"
FORCE_TORCHREC_SOURCE_BUILD="${FORCE_TORCHREC_SOURCE_BUILD:-0}"

usage() {
    cat <<'EOF'
Usage: bash init_dlrm.sh [--mirror=0] [--workspace-root PATH] [--fbgemm-ref REF] [--torchrec-ref REF]

Default behavior:
  - keep the existing global torch install untouched
  - install pinned torchrec/fbgemm-gpu/torchmetrics versions globally

Fallback behavior:
  - set FORCE_TORCHREC_SOURCE_BUILD=1 to build fbgemm_gpu + torchrec from source
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mirror=0)
            MIRROR=0
            ;;
        --workspace-root)
            WORKSPACE_ROOT="$2"
            shift
            ;;
        --fbgemm-ref)
            FBGEMM_REF="$2"
            shift
            ;;
        --torchrec-ref)
            TORCHREC_REF="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

PIP_INDEX_ARGS=()
if [ "$MIRROR" = "1" ]; then
    PIP_INDEX_ARGS=(--index-url https://pypi.tuna.tsinghua.edu.cn/simple)
fi

pip_install() {
    "$PYTHON_BIN" -m pip install "$@" "${PIP_INDEX_ARGS[@]}"
}

user_base_bin() {
    "$PYTHON_BIN" -m site --user-base
}

find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi

    local candidate=""
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /usr/bin/nvcc; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

setup_cuda_toolkit_env() {
    local nvcc_path=""
    nvcc_path="$(find_nvcc || true)"
    if [ -z "${nvcc_path}" ]; then
        return 1
    fi

    local cuda_bin=""
    local cuda_home=""
    cuda_bin="$(dirname "${nvcc_path}")"
    cuda_home="$(cd "${cuda_bin}/.." && pwd)"

    export CUDA_HOME="${cuda_home}"
    export PATH="${cuda_bin}:${PATH}"
    if [ -d "${cuda_home}/lib64" ]; then
        export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
    fi
    return 0
}

ensure_repo() {
    local repo_url="$1"
    local repo_ref="$2"
    local target_dir="$3"

    if [ ! -d "$target_dir/.git" ]; then
        git clone --recursive "$repo_url" "$target_dir"
    fi

    git -C "$target_dir" fetch --tags origin
    git -C "$target_dir" checkout "$repo_ref"
    git -C "$target_dir" submodule sync --recursive
    git -C "$target_dir" submodule update --init --recursive
}

readarray -t TORCH_INFO < <("$PYTHON_BIN" - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda or "cpu")
print(int(torch.compiled_with_cxx11_abi()))
PY
)

TORCH_VERSION="${TORCH_INFO[0]}"
TORCH_CUDA_VERSION="${TORCH_INFO[1]}"
TORCH_CXX11_ABI="${TORCH_INFO[2]}"

echo "Global torch version: ${TORCH_VERSION}"
echo "Global torch cuda: ${TORCH_CUDA_VERSION}"
echo "Global torch cxx11abi: ${TORCH_CXX11_ABI}"

if [ "${TORCH_CUDA_VERSION}" = "cpu" ]; then
    echo "Detected CPU-only torch install; skipping TorchRec/FBGEMM install"
    exit 0
fi

if [ "${FORCE_TORCHREC_SOURCE_BUILD}" = "1" ]; then
    if ! setup_cuda_toolkit_env; then
        echo "FORCE_TORCHREC_SOURCE_BUILD=1 requires CUDA toolkit (nvcc), but nvcc is not available"
        exit 1
    fi
    nvcc --version || true
fi

if [ "${FORCE_TORCHREC_SOURCE_BUILD}" != "1" ]; then
    pip_install --upgrade pip setuptools wheel
    pip_install "torchmetrics==${PINNED_TORCHMETRICS_VERSION}" tqdm
    "$PYTHON_BIN" -m pip install \
        "fbgemm-gpu==${PINNED_FBGEMM_GPU_VERSION}" \
        "torchrec==${PINNED_TORCHREC_VERSION}"

    "$PYTHON_BIN" - <<'PY'
import torch
import fbgemm_gpu
import torchrec

print('Torch version:', torch.__version__)
print('Torch cuda:', torch.version.cuda)
print('Torch cxx11abi:', torch.compiled_with_cxx11_abi())
print('TorchRec import path:', torchrec.__file__)
print('FBGEMM import path:', fbgemm_gpu.__file__)
PY
    exit 0
fi

echo "FORCE_TORCHREC_SOURCE_BUILD=1, building TorchRec stack from source against the current torch install"
mkdir -p "$WORKSPACE_ROOT"
FBGEMM_SRC_DIR="$WORKSPACE_ROOT/FBGEMM"
TORCHREC_SRC_DIR="$WORKSPACE_ROOT/torchrec"

pip_install --upgrade pip setuptools wheel cmake ninja scikit-build packaging "numpy<2" setuptools-git-versioning tabulate
pip_install "torchmetrics==${PINNED_TORCHMETRICS_VERSION}" tqdm iopath pyre-extensions
export PATH="$(user_base_bin)/bin:$PATH"
export USE_CXX11_ABI="${TORCH_CXX11_ABI}"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CXXFLAGS:-}"
"$PYTHON_BIN" -m pip uninstall -y torchrec fbgemm-gpu fbgemm_gpu || true

ensure_repo "$FBGEMM_REPO" "$FBGEMM_REF" "$FBGEMM_SRC_DIR"
ensure_repo "$TORCHREC_REPO" "$TORCHREC_REF" "$TORCHREC_SRC_DIR"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
export MAX_JOBS="${MAX_JOBS:-$CMAKE_BUILD_PARALLEL_LEVEL}"

pushd "$FBGEMM_SRC_DIR/fbgemm_gpu"
rm -rf _skbuild build dist
"$PYTHON_BIN" setup.py bdist_wheel
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps dist/*.whl
popd

pushd "$TORCHREC_SRC_DIR"
"$PYTHON_BIN" -m pip install -e . --no-deps
popd

"$PYTHON_BIN" - <<'PY'
import torch
import fbgemm_gpu
import torchrec

required_ops = [
    'asynchronous_complete_cumsum',
    'bounds_check_indices',
]
missing = [name for name in required_ops if not hasattr(torch.ops.fbgemm, name)]
if missing:
    raise SystemExit(f"Missing registered fbgemm ops after build: {missing}")

print('Torch version:', torch.__version__)
print('TorchRec import path:', torchrec.__file__)
print('FBGEMM import path:', fbgemm_gpu.__file__)
print('Verified fbgemm ops:', ', '.join(required_ops))
PY
