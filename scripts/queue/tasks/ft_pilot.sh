# Item 10 pilot: train variant heads_sf for up to 1000 steps (validation every 100) to choose max_steps for all variants.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/ft_common.sh
mkdir -p $RESULT_DIR
RUN=$FT/runs/pilot; any4d_env; cp $SRC/e5_ft_lib.py $SRC/e5_ft_train.py scripts/
python scripts/e5_ft_train.py --variant heads_sf --data $FT_DATA --e5 $SRC --out $RUN --max_steps ${FT_PILOT_STEPS:-1000} \
  --val_every ${FT_PILOT_VAL:-100} --patience 1000 --export 0 --max_hours 1.5 ${FT_SMOKE_ARGS:-} 2>&1 | { grep -v -i warn || true; }
$POST $SRC/e5_ft_report.py pilot $RUN $RESULT_DIR $TASK_ID $FT/chosen_steps.json
cat $FT/chosen_steps.json
