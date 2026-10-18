# Any4D cross-model reference on exactly the original 100 v2 scenes; results go to the t7_any4d_v2 results folder.
export RESULT_DIR=$QDIR/results/t7_any4d_v2; mkdir -p $RESULT_DIR
MODEL=any4d; source $QDIR/tasks/t7_xm_model.sh
