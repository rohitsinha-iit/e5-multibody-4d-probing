# Item 8b: MPI Sintel training (final and clean passes, every 4th frame t of each sequence, pair t -> t+1),
# Any4D output-level metrics. 2 views (t, t+1) and 4 views (t..t+3); square 518 and wide 518x252 input.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/t8_common.sh
step fetch; bash $SRC/e5_realdata_fetch.sh sintel
any4d_env; cp $SRC/e5_realdata_run.py $SRC/e5_models.py scripts/
step gt; python $SRC/e5_realdata_gt.py sintel $R/sintel $R/gt/sintel --stride 4 --procs 12 ${T8_LIMIT:+--limit $T8_LIMIT}
$POST $SRC/e5_realdata_gtstats.py $R/gt/sintel $RESULT_DIR/gt_stats_sintel.csv
for ps in final clean; do
  for cfg in "2v 0,1" "mv4 0,1,2,3"; do set -- $cfg
    for rz in square wide; do step "sintel $ps $1 $rz"; run_cfg sintel $ps $1 $2 $rz; done
  done
done
python $SRC/e5_realdata_summary.py $RESULT_DIR/summary_sintel $RESULT_DIR/sintel_*.csv
$POST $SRC/e5_realdata_viz.py $R/gt/sintel final $RESULT_DIR/gt_check_sintel.png 6 || true
