# Round 2, review item 1 (W2): CoTracker3 and RAFT on the Kubric v2 (out100) and floating (v3float) scenes, scored on
# the same query sets as Any4D and VGGT (e5_xm_eval.py). Any4D and VGGT v2 predictions are reused from round 1
# ($BIG/xmodel); on v3float both are run here (Any4D in the xm format, VGGT with the tracker_fixed head, no internals).
# Every step is resumable (per-scene npz / json). W2_LIMIT=n restricts to the first n scenes; W2_ROOT moves the outputs.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/w2_common.sh
w2_setup
step "any4d on v3float (xm format)"
( any4d_env; w2_gpu; cp $SRC/e5_xm_any4d.py scripts/; python scripts/e5_xm_any4d.py --data $V3F --out $(pred_dir any4d v3float) --e5 $SRC $W2LIM 2>&1 | grep -v -i warn | tail -2 )
step "vggt on v3float (tracker_fixed head, no internals)"
( source $XMV/bin/activate; w2_gpu; cd $SRC
  CK=$(grep -E "VGGT-1B.*model.pt" $XM/vggt/weights.log | tail -1); TK=$(grep -E "VGGT_tracker_fixed.*e20.pt" $XM/vggt/weights.log | tail -1)
  python e5_xm_vggt.py --data $V3F --out $W2/pred/vggt_v3float --ckpt "$CK" --track_ckpt "$TK" --no_internals $W2LIM 2>&1 | grep -v -i warn | tail -2 )
for ds in v2 v3float; do
  D=$(ds_path $ds)
  ( source $XMV/bin/activate; w2_gpu; cd $SRC
    step "cotracker3 24 frames, $ds"; python e5_w2_predict.py --model cotracker3 --frames all --data $D --out $(pred_dir cotracker3_24f $ds) --e5 $SRC --cotracker_src $CTS --cotracker_ckpt $CTCK $W2LIM 2>&1 | tail -1
    step "cotracker3 7 frames, $ds"; python e5_w2_predict.py --model cotracker3 --frames 0,4,8,12,16,20,23 --data $D --out $(pred_dir cotracker3_7f $ds) --e5 $SRC --cotracker_src $CTS --cotracker_ckpt $CTCK $W2LIM 2>&1 | tail -1
    step "raft direct 0 to 23, $ds"; python e5_w2_predict.py --model raft --raft_mode direct --frames 0,23 --data $D --out $(pred_dir raft_direct $ds) --e5 $SRC $W2LIM 2>&1 | tail -1
    step "raft chained 7 frames, $ds"; python e5_w2_predict.py --model raft --raft_mode chain --frames 0,4,8,12,16,20,23 --data $D --out $(pred_dir raft_chain7 $ds) --e5 $SRC $W2LIM 2>&1 | tail -1
    step "raft chained 24 frames, $ds"; python e5_w2_predict.py --model raft --raft_mode chain --frames all --data $D --out $(pred_dir raft_chain24 $ds) --e5 $SRC $W2LIM 2>&1 | tail -1 )
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for ds in v2 v3float; do for m in $W2_MODELS; do
  step "eval $m $ds"
  $A4PY $SRC/e5_w2_eval.py --data $(ds_path $ds) --pred $(pred_dir $m $ds) --out $W2/eval/${m}_$ds --model $m --procs 16 $W2LIM \
    --combine $RESULT_DIR/w2_eval_${m}_$ds.json | tail -2
done; done
step "summary"
$A4PY $SRC/e5_w2_analysis.py outputs --kubric $RESULT_DIR --out $RESULT_DIR | tail -20
ls -la $RESULT_DIR
