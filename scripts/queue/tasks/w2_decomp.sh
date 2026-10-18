# Round 2, review item 1 (W2): alpha/beta decomposition (camera-motion pull vs shrinkage toward zero image motion) and
# slope vs distance to the object boundary (spatial smoothing), all models and datasets. CPU only, minutes.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/w2_common.sh
KR=${W2_KUBRIC_DIR:-$QDIR/results/w2_kubric_baselines}; RR=${W2_REAL_DIR:-$QDIR/results/w2_real_baselines}
$A4PY $SRC/e5_w2_analysis.py decomp --kubric $KR --real $RR --out $RESULT_DIR | tail -30
ls -la $RESULT_DIR
