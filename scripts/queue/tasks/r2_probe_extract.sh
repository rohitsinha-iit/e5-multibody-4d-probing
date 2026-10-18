# Round 2, review W4 (plan item 6): control-model internals for the probe controls, v2 out100 and v3float.
# randtf = public DINOv2 + random Any4D transformer; randall = everything random; enc_any4d / enc_dino = raw encoder tokens.
source $QDIR/tasks/common.sh; set -e
PC=${R2_PC:-$BIG/probe_ctrl}   # R2_PC, R2_DS, R2_SEEDS, R2_LIMIT: smoke-test overrides only
DSL=${R2_DS:-"v2 v3float"}; SEEDS=${R2_SEEDS:-"0 1 2 3 4"}; LIM=${R2_LIMIT:-0}; mkdir -p $PC/feats
any4d_env
cp $SRC/e5_probe_extract.py scripts/
for DS in $DSL; do
  D=$(ds_dir $DS); DATA=$D/out; [ $DS = v2 ] && DATA=$D/out100
  need $DATA; need $D/gt_patches; F=$PC/feats/$DS; mkdir -p $F
  step "$DS randtf (+ public DINOv2 tokens)"; python scripts/e5_probe_extract.py --data $DATA --gt $D/gt_patches --variant randtf --limit $LIM --out $F/randtf --enc_out $F/enc_dino 2>&1 | grep -v -i warn | tail -2
  step "$DS randall"; python scripts/e5_probe_extract.py --data $DATA --gt $D/gt_patches --variant randall --limit $LIM --out $F/randall 2>&1 | grep -v -i warn | tail -2
  step "$DS trained encoder tokens"; python scripts/e5_probe_extract.py --data $DATA --gt $D/gt_patches --variant trained --limit $LIM --enc_out $F/enc_any4d 2>&1 | grep -v -i warn | tail -2
  step "$DS trained internals (symlink view)"; mkdir -p $F/trained; cp $D/internals/stored_layers.json $F/trained/
  for s in $(ls $DATA | if [ $LIM -gt 0 ]; then head -$LIM; else cat; fi); do
    if [ -f $D/internals/${s}_tok.npy ] && [ -f $D/gt_patches/${s}_gt.npz ]; then ln -sf $D/internals/${s}_tok.npy $F/trained/; fi
  done
  for v in trained randtf randall enc_dino enc_any4d; do echo "$DS $v $(ls $F/$v | grep -c _tok.npy) scenes"; done
done
python $SRC/e5_probe_extract_summary.py $PC/feats $RESULT_DIR $TASK_ID
du -sh $PC/feats/*/* | tee $RESULT_DIR/disk_usage.txt
