#!/bin/bash
# U5 all-view motion probe on Any4D (fit scenes; L* by CV; 90% recall threshold)
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
$PY $A/motion_probe_allviews.py --mode all --split fit
