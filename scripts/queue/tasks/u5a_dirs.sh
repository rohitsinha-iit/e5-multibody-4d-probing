#!/bin/bash
# U5a counterfactual directions (P_obj at k* + five E5 heads) on fit scenes
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
K=$($PY -c "import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['obj']['kstar'] or 0)"); $PY $A/repair.py --mode dirs --split fit --kstar_obj $K
