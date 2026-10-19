#!/bin/bash
# Gate G0h: VGGT camera-token layout (identity and camera-head positive check)
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=12; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/xm/bin/python
$PY $A/vggt_hooks.py
