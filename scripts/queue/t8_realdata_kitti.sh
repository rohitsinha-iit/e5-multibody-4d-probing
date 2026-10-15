# Item 8a: KITTI scene flow 2015 (200 training scenes), Any4D output-level metrics (slope, EPE, sep_ratio).
# 2 views (frames 10, 11), 4 views (10..13), 7 views (10..16); square 518 (as on Kubric) and wide 518x168 input.
# Resumable: fetch, GT and predictions are cached on the big disk; metric rows are checkpointed per scene.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/t8_common.sh
step fetch; bash $SRC/e5_realdata_fetch.sh kitti kitti_mv
any4d_env; cp $SRC/e5_realdata_run.py $SRC/e5_models.py scripts/
step gt; python $SRC/e5_realdata_gt.py kitti15 $R/kitti15 $R/gt/kitti15 --procs 12 ${T8_LIMIT:+--limit $T8_LIMIT}
$POST $SRC/e5_realdata_gtstats.py $R/gt/kitti15 $RESULT_DIR/gt_stats_kitti15.csv
for cfg in "2v 0,1" "mv4 0,1,2,3" "mv7 0,1,2,3,4,5,6"; do set -- $cfg
  for rz in square wide; do step "kitti15 $1 $rz"; run_cfg kitti15 rgb $1 $2 $rz; done
done
python $SRC/e5_realdata_summary.py $RESULT_DIR/summary_kitti15 $RESULT_DIR/kitti15_*.csv
$POST $SRC/e5_realdata_viz.py $R/gt/kitti15 rgb $RESULT_DIR/gt_check_kitti15.png 6 || true
