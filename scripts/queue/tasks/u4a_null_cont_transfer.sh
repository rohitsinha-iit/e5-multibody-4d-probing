#!/bin/bash
# U4a joint top-k with 40 nulls on v3float transfer (cont)
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
$PY $A/interchange.py --mode joint --screen cont --split transfer --allow_test --n_null 40
