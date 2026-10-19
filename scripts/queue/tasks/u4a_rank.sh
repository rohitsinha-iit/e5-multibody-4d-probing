#!/bin/bash
# U4a rank components on fit scenes (obj, cam, cont, camwarp)
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=0; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
for s in obj cam cont camwarp objall; do $PY $A/interchange.py --mode rank --screen $s --split fit; done
