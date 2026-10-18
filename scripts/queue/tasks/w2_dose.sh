# Round 2, review item 1 (W2): dose-response of the per-object slope (speed, area, texture, occlusion, camera rotation,
# number of motions, camera-induced displacement, model confidence) and a standardised regression. CPU only.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/w2_common.sh
KR=${W2_KUBRIC_DIR:-$QDIR/results/w2_kubric_baselines}
export OMP_NUM_THREADS=8
$A4PY $SRC/e5_w2_analysis.py dose --kubric $KR --out $RESULT_DIR | tail -30
ls -la $RESULT_DIR
