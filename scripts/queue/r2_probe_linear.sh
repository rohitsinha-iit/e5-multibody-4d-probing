# Round 2, review W4: linear probes (part P of e5_internal_analysis.py, round-1 definitions) on the trained model,
# random-init controls and raw encoder tokens; 5 probe seeds; label-shuffle controls (patch, instance). v2 out100 and v3float.
source $QDIR/tasks/common.sh; set -e
PC=${R2_PC:-$BIG/probe_ctrl}   # R2_PC, R2_DS, R2_SEEDS, R2_LIMIT: smoke-test overrides only
DSL=${R2_DS:-"v2 v3float"}; SEEDS=${R2_SEEDS:-"0 1 2 3 4"}; LIM=${R2_LIMIT:-0}
any4d_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LAY=0,4,8,12,16,20,24
run() {  # run DS SOURCE MODE SEED LAYERS
  local DS=$1 S=$2 M=$3 SEED=$4 L=$5 O=$PC/probes/$1/$2/lin_$3_s$4 SH=""
  test -f ${O}_results.json && return 0
  mkdir -p $PC/probes/$DS/$S; [ $M != real ] && SH=$M
  if E5_SHUFFLE=$SH E5_PARTS=P E5_LAYERS=$L python $SRC/e5_internal_analysis.py $PC/feats/$DS/$S $(ds_dir $DS)/gt_patches ${O}.tmp > ${O}.log 2>&1; then
    mv ${O}.tmp_results.json ${O}_results.json; echo "$DS $S $M s$SEED ok"
  else echo "$DS $S $M s$SEED FAILED"; tail -3 ${O}.log; return 1; fi
}
for DS in $DSL; do for v in trained randtf randall enc_dino enc_any4d; do need $PC/feats/$DS/$v/stored_layers.json; done; done
# priority order: real labels first, then the shuffled controls
for M in real patch instance; do
  for DS in $DSL; do
    for SEED in $SEEDS; do
      step "$DS $M seed $SEED"
      for S in trained randall randtf; do run $DS $S $M $SEED $LAY; done
      for S in enc_any4d enc_dino; do run $DS $S $M $SEED 0; done
    done
  done
done
python $SRC/e5_probe_summary.py $PC/probes $RESULT_DIR
{ echo "# $TASK_ID: linear probe controls (review W4, plan item 6)"; echo
  echo "Linear probes are part P of e5_internal_analysis.py with the round-1 definitions (ridge moving-vs-static probe on the view-0 token, low-rank bilinear same-rigid-body pair probe on concat(view 0, view 6), 5-fold CV grouped by scene), now seeded (--seed 0..4; seeds change probe init and pair sampling). Sources: Any4D trained (layers 0,4,...,24), random init everything (randall), public DINOv2 + random transformer (randtf), raw encoder tokens without view PE (enc_any4d = trained Any4D encoder, enc_dino = public DINOv2). Controls: E5_SHUFFLE=patch permutes all patch labels within a scene (counts kept); E5_SHUFFLE=instance permutes body labels across object instances within a scene (a type-level control: instance identity is kept). Selectivity = real minus control. This task has only the linear probes; the MLP probes and the combined figure are in r2_probe_mlp."; echo
  cat $RESULT_DIR/probe_controls_key.md; } > $RESULT_DIR/${TASK_ID}_summary.md
ls -la $RESULT_DIR
