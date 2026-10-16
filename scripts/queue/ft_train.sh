# Item 10: train one fine-tuning variant ($1). Steps from the pilot ($FT/chosen_steps.json), early stopping on the
# held-out 10% of the training scenes, at most 3.5 h. Resumable (last.pt every 200 steps and at every validation).
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/ft_common.sh
mkdir -p $RESULT_DIR
V=$1; RUN=$FT/runs/$V; need $FT/chosen_steps.json
any4d_env; cp $SRC/e5_ft_lib.py $SRC/e5_ft_train.py scripts/
VE=$(python -c "import json;n=json.load(open('$FT/chosen_steps.json'))['max_steps'];print(max(100,int(round(n/10,-2))))")
python scripts/e5_ft_train.py --variant $V --data $FT_DATA --e5 $SRC --out $RUN --steps_file $FT/chosen_steps.json \
  --val_every $VE --patience 4 --max_hours 3.5 ${FT_SMOKE_ARGS:-} 2>&1 | { grep -v -i warn || true; }
need $RUN/any4d_ft.pth
$POST $SRC/e5_ft_report.py train $RUN $RESULT_DIR $TASK_ID
ls -la $RUN
