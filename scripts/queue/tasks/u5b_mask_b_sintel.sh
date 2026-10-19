#!/bin/bash
# U5b masking variant b on sintel
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
KC=$($PY -c "import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['cont']['kstar'] or 5)"); $PY $A/repair.py --mode u5b --variant b --kstar_cont $KC --eval sintel --allow_test
