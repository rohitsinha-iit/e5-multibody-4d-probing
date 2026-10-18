# Item 10: evaluate one model ($1 = base or a variant) with the round-1 metric code.
# Kubric: v2 out100, v3float, v3far (7 views 0,4,..,20,23; target frame 23): slope, EPE, sep_ratio, mask IoU/AUC,
#         depth and pose error vs GT. Internals: layers 0, 17, 24 on v3float, probe P (motion AUC).
# Real data (wide input): KITTI 2015 2 and 4 views, Sintel final 2 and 4 views, Sintel clean 2 views.
# FT_LIMIT=N restricts every dataset to N scenes (smoke test). Resumable: predictions and rows are cached.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/ft_common.sh; source $QDIR/tasks/t8_common.sh
mkdir -p $RESULT_DIR
V=$1; CK=$(ft_ckpt $V); need $CK; O=$FT/eval/$V${FT_LIMIT:+_smoke}; mkdir -p $O
any4d_env; cp $SRC/e5_ft_lib.py scripts/; python $SRC/e5_ft_patch.py $SRC scripts
subset() {  # subset DATA_DIR NAME N -> dir with symlinks to the first N scenes (scene order is kept)
  local S=$O/subset_$2; mkdir -p $S
  ls -d $1/*/ | head -$3 | while read d; do ln -sfn $(readlink -f $d) $S/$(basename $d); done; echo $S
}
for ds in v2 v3float v3far; do
  D=$(ft_eval_data $ds); [ -n "$FT_LIMIT" ] && D=$(subset $D $ds $FT_LIMIT)
  step "$V $ds inference"; quiet python scripts/e5_any4d_infer.py --data $D --out $O/pred_$ds --checkpoint_path $CK | tail -2
  step "$V $ds metrics"; python $SRC/e5_ft_eval_kubric.py --data $(ft_eval_data $ds) --pred $O/pred_$ds --out $RESULT_DIR/kubric_$ds.json --src $SRC
done
D=$BIG/data/v3float/out; [ -n "$FT_LIMIT" ] && D=$(subset $D v3float_int $((FT_LIMIT > 6 ? FT_LIMIT : 6)))
step "$V internals (v3float, layers 0,17,24)"
E5_CKPT=$CK quiet python scripts/e5_ft_extract.py --data $D --out $O/internals_v3float --layers 0,17,24 | tail -1
cp $SRC/e5_internal_analysis.py $SRC/e5_multibody.py $O/
test -f $RESULT_DIR/probeP_v3float_results.json || (cd $O && E5_PARTS=P E5_LAYERS=0,17,24 OMP_NUM_THREADS=8 \
  python e5_internal_analysis.py internals_v3float $BIG/data/v3float/gt_patches $RESULT_DIR/probeP_v3float > probeP.log 2>&1; tail -2 probeP.log)
for cfg in "kitti15 rgb 2v 0,1" "kitti15 rgb mv4 0,1,2,3" "sintel final 2v 0,1" "sintel final mv4 0,1,2,3" "sintel clean 2v 0,1"; do
  set -- $cfg; step "$V real $1 $2 $3"
  quiet python scripts/e5_realdata_run.py --src $SRC --gt $R/gt/$1 --pass $2 --offsets $4 --resize wide --checkpoint $CK \
    --pred $O/pred_real/${1}_${2}_${3}_wide --out $RESULT_DIR/${1}_${2}_${3}_wide.csv ${FT_LIMIT:+--limit $FT_LIMIT} | tail -1
done
$POST $SRC/e5_ft_report.py eval $RESULT_DIR $TASK_ID $V
du -sh $O
