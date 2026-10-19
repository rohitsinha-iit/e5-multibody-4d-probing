#!/bin/bash
# U4a secondary warp-aligned camera screen (24 residual states) on fit scenes
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB=14; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
$PY $A/interchange.py --mode screen --screen camwarp --split fit --comps b0r,b1r,b2r,b3r,b4r,b5r,b6r,b7r,b8r,b9r,b10r,b11r,b12r,b13r,b14r,b15r,b16r,b17r,b18r,b19r,b20r,b21r,b22r,b23r
