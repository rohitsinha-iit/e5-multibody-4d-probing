# Item 3: number of input views (2, 3, 4; 7 exists) on dataset $1 (default v2). Outputs, probes and attention.
source $QDIR/tasks/common.sh; set -e
DS=${1:-v2}; SRCD=$(ds_dir $DS); need $SRCD/out
D=$BIG/viewsweep/$DS; mkdir -p $D; ln -sfn $SRCD/out $D/out
# the 7-view run reuses the dataset's own predictions and internals
ln -sfn $SRCD/pred_any4d_7v $D/pred_any4d_7v; test -d $SRCD/internals && ln -sfn $SRCD/internals $D/internals_full
cp $SRC/e5_step3_analysis.py $SRC/e5_gt_patches.py $SRC/e5_internal_analysis.py $SRC/e5_multibody.py $SRC/e5_kubric_post.py $D/
LAY=0,4,8,12,16,20,24
declare -A FR=( [2]=0,23 [3]=0,12,23 [4]=0,8,16,23 [7]=0,4,8,12,16,20,23 )
for NV in 2 3 4 7; do
  any4d_env
  step "$DS ${NV} views: inference"
  python scripts/e5_any4d_infer.py --data $D/out --out $D/pred_any4d_${NV}v --frames ${FR[$NV]} 2>&1 | grep -v -i warn | tail -2
  if [ $NV != 7 ] || [ ! -d $D/internals_full ]; then
    step "$DS ${NV} views: extraction (layers $LAY)"
    python scripts/e5_any4d_extract.py --data $D/out --out $D/internals_${NV}v --frames ${FR[$NV]} --layers $LAY 2>&1 | grep -v -i warn | tail -2
    INT=$D/internals_${NV}v
  else
    INT=$D/internals_full
  fi
  cd $D
  step "$DS ${NV} views: step3 + gt patches"
  test -f $RESULT_DIR/step3_${DS}_${NV}v.json || python e5_step3_analysis.py out pred_any4d_${NV}v $RESULT_DIR/step3_${DS}_${NV}v.json > step3_${NV}v.log 2>&1
  python e5_gt_patches.py out pred_any4d_${NV}v gt_patches_${NV}v > gt_${NV}v.log 2>&1
  step "$DS ${NV} views: probes + attention"
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  test -f $RESULT_DIR/vs_${DS}_${NV}v_results.json || \
    E5_PARTS=A,P E5_LAYERS=$LAY python e5_internal_analysis.py $INT gt_patches_${NV}v $RESULT_DIR/vs_${DS}_${NV}v > vs_${NV}v.log 2>&1
  tail -2 vs_${NV}v.log
done
ls -la $RESULT_DIR
