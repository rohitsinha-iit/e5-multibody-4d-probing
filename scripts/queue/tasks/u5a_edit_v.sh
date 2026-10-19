#!/bin/bash
# U5a variant v: heads_sel LoRA, test and transfer
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=16; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
for sp in test transfer; do $PY $A/repair.py --mode u5a --variant v --split $sp --allow_test; done
