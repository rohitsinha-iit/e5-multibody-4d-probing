# Item 7: cross-model comparison table across Any4D, MapAnything, VGGT-1B and PAGE-4D.
source $QDIR/tasks/common.sh; set -e
$POST $SRC/e5_xm_summary.py $QDIR/results $RESULT_DIR/xm_comparison
