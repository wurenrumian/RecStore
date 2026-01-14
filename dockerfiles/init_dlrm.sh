#!/bin/bash
cd "$(dirname "$0")"
set -x
set -e

pip install fbgemm-gpu==1.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchmetrics==1.0.3
pip install torchrec==1.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Usage
# torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py