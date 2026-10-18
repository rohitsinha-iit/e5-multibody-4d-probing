# Review item W3 at scale (after visual approval, gate v4_visual): 30 scenes per new split (120 scenes), GT checks.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh; source $QDIR/tasks/v4_common.sh
step "rendering v4 epipolar set (resumable)"; v4_render_list ALL
D=$(ds_dir $V4DS)
step "post"; post_ds $D
cp $SRC/e5_v4_epicheck.py $SRC/e5_step3_analysis.py $D/
step "GT checks"; (cd $D && $POST e5_v4_epicheck.py out $RESULT_DIR/v4_scale_epi "v4_scale_epi: GT checks of the v4 epipolar set (120 scenes)" > epicheck.log 2>&1; tail -2 epicheck.log)
cp $D/scene_summary.json $RESULT_DIR/
test -f $RESULT_DIR/v4_scale_epi_summary.md || { echo "GT check failed"; tail -20 $D/epicheck.log; exit 1; }
N=$(ls $D/out | wc -l); echo "v4 scenes: $N"
test $N -ge 112 || { echo "too few scenes rendered ($N)"; exit 1; }
