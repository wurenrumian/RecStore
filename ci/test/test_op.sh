#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIB_PATH="${REPO_ROOT}/build/lib/lib_recstore_ops.so"
PY_CLIENT_DIR="${REPO_ROOT}/src/framework/pytorch/python_client"
PY_PKG_ROOT="${REPO_ROOT}/src/python/pytorch"

if [ ! -f "${LIB_PATH}" ]; then
    echo "${LIB_PATH} does not exist."
    exit 1
fi

if [[ "$@" != *"--mock"* ]]; then
  echo "Not in mock mode; skipping Python execution."
  exit 0
fi

export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PY_PKG_ROOT}:${PYTHONPATH:-}"

if [[ "${CI:-false}" == "true" ]] || [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
  echo "CI environment detected: tests will assume ps_server is already running"
fi

echo "[1/2] Running pytorch_client_test"
(
  cd "${PY_CLIENT_DIR}"
  python3 client_test.py "${LIB_PATH}"
)

echo "[2/2] Running dist_emb_unittest"
(
  cd "${PY_PKG_ROOT}"
  python3 -m unittest recstore.unittest.test_dist_emb
)

# echo "[3/3] Running ebc_precision_unittest"
# (
#   cd "${PY_PKG_ROOT}"
#   python3 -m unittest recstore.unittest.test_ebc_precision_wrapper
# )

echo "All Python tests finished successfully."