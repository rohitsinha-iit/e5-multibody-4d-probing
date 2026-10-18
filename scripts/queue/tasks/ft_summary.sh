# Item 10: comparison table and figure over base and all variants (paired by scene, bootstrap CIs, sign tests).
source $QDIR/tasks/common.sh; set -e
mkdir -p $RESULT_DIR
$POST $SRC/e5_ft_summary.py ${FT_QR:-$QDIR/results} $RESULT_DIR $TASK_ID
ls -la $RESULT_DIR
