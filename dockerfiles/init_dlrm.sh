#!/bin/bash
cd "$(dirname "$0")"
set -x
set -e

MIRROR=1
for arg in "$@"; do
	case $arg in
		--mirror=0)
			MIRROR=0
			;;
	esac
done

if [ "$MIRROR" = "0" ]; then
	PIP_MIRROR=""
else
	PIP_MIRROR="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
fi

pip install fbgemm-gpu==1.0 $PIP_MIRROR
pip install torchmetrics==1.0.3 $PIP_MIRROR
pip install torchrec==1.0 $PIP_MIRROR

# Usage
# bash init_dlrm.sh [--mirror=0]
# torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py