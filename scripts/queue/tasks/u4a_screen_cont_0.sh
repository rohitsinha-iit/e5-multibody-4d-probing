#!/bin/bash
# U4a cont screen on fit scenes, component chunk 0/4
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
step u4a_screen_cont_0; $PY $A/interchange.py --mode screen --screen cont --split fit --chunk 0/4
