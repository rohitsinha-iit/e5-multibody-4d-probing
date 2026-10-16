# Round 2, review W4: nonlinear probes (2-layer MLP, 256 hidden, AdamW weight decay, early stopping, grouped CV by scene)
# and a matched linear probe with the same recipe; 5 seeds; patch and instance label-shuffle controls. v2 out100 and v3float.
# Writes the combined (linear + MLP) summary and figure.
source $QDIR/tasks/common.sh; set -e
PC=${R2_PC:-$BIG/probe_ctrl}   # R2_PC, R2_DS, R2_SEEDS, R2_LIMIT: smoke-test overrides only
DSL=${R2_DS:-"v2 v3float"}; SEEDS=${R2_SEEDS:-"0 1 2 3 4"}; LIM=${R2_LIMIT:-0}
any4d_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ALL=0,4,8,12,16,20,24
run() {  # run DS SOURCE MODE SEED LAYERS
  local DS=$1 S=$2 M=$3 SEED=$4 L=$5 O=$PC/probes/$1/$2/mlp_$3_s$4.json SH=""
  test -f $O && return 0
  mkdir -p $PC/probes/$DS/$S; [ $M != real ] && SH="--shuffle $M"
  if python $SRC/e5_probe_mlp.py $PC/feats/$DS/$S $(ds_dir $DS)/gt_patches $O --layers $L --seed $SEED $SH > ${O%.json}.log 2>&1; then
    echo "$DS $S $M s$SEED ok"
  else echo "$DS $S $M s$SEED FAILED"; tail -3 ${O%.json}.log; return 1; fi
}
for DS in $DSL; do for v in trained randtf randall enc_dino enc_any4d; do need $PC/feats/$DS/$v/stored_layers.json; done; done
for M in real patch instance; do
  for DS in $DSL; do
    for SEED in $SEEDS; do
      step "$DS $M seed $SEED"
      run $DS trained $M $SEED $ALL
      if [ $M = real ]; then RL=$ALL; else RL=0,24; fi   # random-init controls: all layers for real labels, layers 0 and 24 for shuffled labels
      for S in randall randtf; do run $DS $S $M $SEED $RL; done
      for S in enc_any4d enc_dino; do run $DS $S $M $SEED 0; done
    done
  done
done
python $SRC/e5_probe_summary.py $PC/probes $RESULT_DIR
{ echo "# $TASK_ID: probe controls, linear and MLP (review W4, plan item 6)"; echo
  echo "MLP probe: 2 layers, 256 hidden, ReLU; motion probe D to 256 to 1 on the view-0 token; pair probe embeds concat(view 0, view 6) with D to 256 to 64 and scores pairs by a scaled dot product (the nonlinear version of the round-1 bilinear probe). AdamW (lr 1e-3, weight decay 1e-2), minibatch 4096, early stopping on a held-out 15% of the training scenes (grouped by scene), same 5 scene folds as round 1. 'linear (MLP recipe)' is the same training with the hidden layer removed. 'linear (round-1 ridge / bilinear)' comes from r2_probe_linear. Five seeds per cell. Shuffled controls: all layers for the trained model, layers 0 and 24 for the random-init variants. Selectivity = real minus control (paired over scenes)."; echo
  echo "Figure: fig_probe_controls.png (object patches: motion AUC and pair AUC), fig_probe_controls_allpatches.png (all pure patches)."; echo
  cat $RESULT_DIR/probe_controls_key.md; } > $RESULT_DIR/${TASK_ID}_summary.md
ls -la $RESULT_DIR
