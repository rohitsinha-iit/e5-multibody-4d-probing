# Item 7 (cross-model): one model on the 100 v2 scenes. Sourced by t7_<model>_v2.sh with MODEL set.
# Steps (each resumable): env + weights, inference with internals, output metrics, per-model GT patches, probes P + attention A.
# XM_LIMIT=n restricts everything to the first n scenes (smoke tests); XM_OUT overrides the output root.
source $QDIR/tasks/common.sh; source $QDIR/tasks/xm_common.sh; set -e
LIM=${XM_LIMIT:+--limit $XM_LIMIT}
OUT=${XM_OUT:-$XM}/$MODEL; mkdir -p $OUT
subset_dir() {  # subset_dir SRC DST N: symlink the first N scenes' internals (all files when N is empty)
  local S=$1 D=$2 N=$3; mkdir -p $D; cp $S/*.json $D/ 2>/dev/null || true
  ls $S | grep '_tok.npy$' | sort | head -n ${N:-100000} | while read f; do b=${f%_tok.npy}; ln -sfn $S/${b}_tok.npy $D/; ln -sfn $S/${b}_att.npz $D/; done
}
if [ $MODEL = any4d ]; then
  any4d_env; cp $SRC/e5_xm_any4d.py scripts/
  step "any4d: inference in the cross-model format"
  python scripts/e5_xm_any4d.py --data $V2/out100 --out $OUT/pred --e5 $SRC $LIM 2>&1 | grep -v -i warn | tail -3
  INT=$V2/internals; GTP=$V2/gt_patches; LAY=0,4,8,12,16,20,24
else
  xm_setup; xm_weights > $OUT/weights.log 2>&1 || { tail $OUT/weights.log; exit 4; }
  CK=$(grep -E "$CKPAT" $OUT/weights.log | tail -1); echo "checkpoint $CK"
  XARGS=""; if [ -n "$TRKPAT" ]; then XARGS="--track_ckpt $(grep -E "$TRKPAT" $OUT/weights.log | tail -1)"; echo "track head: $XARGS"; fi
  xm_env; cd $SRC
  step "$MODEL: inference + internals"
  python $SRC/$RUNNER --data $V2/out100 --out $OUT --ckpt "$CK" $XARGS $LIM 2>&1 | grep -v -i warn | tail -4
  INT=$OUT/internals; GTP=$OUT/gt_patches; LAY=$XLAY
  python $SRC/e5_xm_gtpatch.py $V2/gt_patches $OUT/pred $GTP
  deactivate
  cd $A4 && source .venv/bin/activate
fi
step "$MODEL: output metrics"
cd $SRC && python e5_xm_eval.py --data $V2/out100 --pred $OUT/pred --out $RESULT_DIR/xm_output_${MODEL}.json --model $MODEL $LIM | tail -2
step "$MODEL: probes (P) + attention (A), layers $LAY"
if [ -n "$XM_LIMIT" ]; then rm -rf $OUT/int_subset; subset_dir $INT $OUT/int_subset $XM_LIMIT; INT=$OUT/int_subset; fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
test -f $RESULT_DIR/xm_internal_${MODEL}_results.json || \
  E5_PARTS=A,P E5_LAYERS=$LAY python $SRC/e5_internal_analysis.py $INT $GTP $RESULT_DIR/xm_internal_${MODEL} > $OUT/internal.log 2>&1 || { tail -30 $OUT/internal.log; exit 5; }
tail -2 $OUT/internal.log
cp $OUT/internals/grid.json $RESULT_DIR/grid_${MODEL}.json 2>/dev/null || true
ls -la $RESULT_DIR
