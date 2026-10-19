#!/bin/bash
# U5a edit variant ii, g=1.5, test and transfer
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
KO=$($PY -c "import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['obj']['kstar'] or 0)"); for sp in test transfer; do $PY $A/repair.py --mode u5a --variant ii --g 1.5 --kstar_obj $KO --split $sp --allow_test; done
