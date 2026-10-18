# Item 10: render the float half of the fine-tuning training set (200 scenes, approved v3float floating shadowless settings, seeds 2201-2400).
# No new scene type, so no visual gate. Resumable: finished scenes are skipped.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh; source $QDIR/tasks/ft_common.sh
mkdir -p $RESULT_DIR
step "render float training scenes"; ft_render float
D=$FT_FLOAT; step "post-processing (flow check, SNR, contact sheets)"; post_ds $D
N=$(ls -d $D/out/*/e5_meta.json 2>/dev/null | wc -l); EXP=${FT_RENDER_LIMIT:-200}
$POST $SRC/e5_ft_report.py render $D $RESULT_DIR $TASK_ID $EXP
cp $D/scene_summary.json $RESULT_DIR/ 2>/dev/null || true
echo "rendered $N of $EXP"; du -sh $D
[ $N -ge $((EXP * 9 / 10)) ]
