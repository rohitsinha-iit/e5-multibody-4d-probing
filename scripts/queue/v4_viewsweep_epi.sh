# Review item W3: view-count sweep (2, 3, 4, 7 views) on the v4 epipolar set pooled with v3float.
# Outputs (step 3, slope/EPE/mask IoU per target), probes (pooled CV and transfer from v3float), attention, summary.
# Env (for smoke tests only): V4_VS (sweep dir name), V4_REF_LIST (subset of v3float scenes), V4_VIEWS, V4_NEW_LIST.
source $QDIR/tasks/common.sh; set -e
NEWD=$(ds_dir v4epi); REFD=$(ds_dir v3float); REFVS=$BIG/viewsweep/v3float
need $NEWD/out; need $REFD/out
D=$BIG/viewsweep/${V4_VS:-v4epi}; mkdir -p $D/out $D/out_new
VIEWS=${V4_VIEWS:-2 3 4 7}
REF_RE='^(epi_aligned|n2_generic|n3_generic|confound_same|confound_glued)_s'
# scene links (complete scenes only)
for s in $NEWD/out/*; do n=$(basename $s)
  [ -n "${V4_NEW_LIST:-}" ] && ! echo " $V4_NEW_LIST " | grep -q " $n " && continue
  test -f $s/e5_meta.json && ln -sfn $s $D/out/$n && ln -sfn $s $D/out_new/$n; done
for s in $REFD/out/*; do n=$(basename $s)
  [ -n "${V4_REF_LIST:-}" ] && ! echo " $V4_REF_LIST " | grep -q " $n " && continue
  test -f $s/e5_meta.json && ln -sfn $s $D/out/$n; done
echo "scenes: $(ls $D/out | wc -l) (new $(ls $D/out_new | wc -l))"
cp $SRC/e5_step3_analysis.py $SRC/e5_gt_patches.py $SRC/e5_internal_analysis.py $SRC/e5_multibody.py $SRC/e5_kubric_post.py \
   $SRC/e5_v4_outmetrics.py $SRC/e5_v4_summary.py $D/
LAY=0,4,8,12,16,20,24
declare -A FR=( [2]=0,23 [3]=0,12,23 [4]=0,8,16,23 [7]=0,4,8,12,16,20,23 )
for NV in $VIEWS; do
  P=$D/pred_any4d_${NV}v; I=$D/internals_${NV}v; G=$D/gt_patches_${NV}v; mkdir -p $P $I $G $D/pred_new_${NV}v
  if [ $NV = 7 ]; then RP=$REFD/pred_any4d_7v; RI=; else RP=$REFVS/pred_any4d_${NV}v; RI=$REFVS/internals_${NV}v; fi
  RG=$REFVS/gt_patches_${NV}v
  # reuse round-1 v3float predictions, internals (same layers) and GT patches by symlink; never write through a link
  for s in $D/out/*; do n=$(basename $s); test -e $NEWD/out/$n && continue
    if [ -f $RP/$n.npz ] && [ ! -e $P/$n.npz ]; then ln -s $RP/$n.npz $P/$n.npz; fi
    if [ -n "$RI" ] && [ -f $RI/${n}_tok.npy ] && [ -f $RI/${n}_att.npz ] && [ ! -e $I/${n}_att.npz ]; then
      ln -sfn $RI/${n}_tok.npy $I/${n}_tok.npy; ln -s $RI/${n}_att.npz $I/${n}_att.npz; fi
    if [ -L $P/$n.npz ] && [ -f $RG/${n}_gt.npz ] && [ ! -e $G/${n}_gt.npz ]; then ln -s $RG/${n}_gt.npz $G/${n}_gt.npz; fi
  done
  any4d_env
  step "${NV} views: inference (new scenes; reference reused)"
  python scripts/e5_any4d_infer.py --data $D/out --out $P --frames ${FR[$NV]} 2>&1 | grep -v -i warn | tail -2
  step "${NV} views: extraction (layers $LAY)"
  python scripts/e5_any4d_extract.py --data $D/out --out $I --frames ${FR[$NV]} --layers $LAY 2>&1 | grep -v -i warn | tail -2
  cd $D
  for s in out_new/*; do n=$(basename $s); test -f $P/$n.npz && ln -sfn $P/$n.npz pred_new_${NV}v/$n.npz; done
  step "${NV} views: gt patches (skip existing), step 3 (new scenes), output metrics (epi scenes)"
  E5_SKIP_EXISTING=1 python e5_gt_patches.py out $P $G > gt_${NV}v.log 2>&1
  test -f $RESULT_DIR/step3_v4epi_${NV}v.json || python e5_step3_analysis.py out_new pred_new_${NV}v $RESULT_DIR/step3_v4epi_${NV}v.json > step3_${NV}v.log 2>&1
  test -f $RESULT_DIR/outm_${NV}v.json || OMP_NUM_THREADS=8 python e5_v4_outmetrics.py out $P $RESULT_DIR/outm_${NV}v.json epi > outm_${NV}v.log 2>&1
  test -f $QDIR/results/v3_viewsweep/step3_v3float_${NV}v.json && cp $QDIR/results/v3_viewsweep/step3_v3float_${NV}v.json $RESULT_DIR/ref_step3_v3float_${NV}v.json || true
  step "${NV} views: probes (pooled CV, paired scenes share a fold) + attention"
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  test -f $RESULT_DIR/vs_pool_${NV}v_results.json || \
    E5_FOLD_BY_SEED=1 E5_PARTS=A,P E5_LAYERS=$LAY python e5_internal_analysis.py $I $G $RESULT_DIR/vs_pool_${NV}v > vs_pool_${NV}v.log 2>&1
  step "${NV} views: probes trained on v3float only (transfer)"
  test -f $RESULT_DIR/vs_xfer_${NV}v_results.json || \
    E5_TRAIN_FILTER="$REF_RE" E5_P_MOTION_ONLY=1 E5_PARTS=P E5_LAYERS=$LAY python e5_internal_analysis.py $I $G $RESULT_DIR/vs_xfer_${NV}v > vs_xfer_${NV}v.log 2>&1
  tail -1 vs_pool_${NV}v.log; tail -1 vs_xfer_${NV}v.log
done
step "summary"
VL=$(echo $VIEWS | tr ' ' ',')
$POST $D/e5_v4_summary.py $RESULT_DIR $D/out ${TASK_ID} $VL
du -sh $D 2>/dev/null || true
ls -la $RESULT_DIR
