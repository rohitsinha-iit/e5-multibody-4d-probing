# Round 2, review item 1 (W2): RAFT (C_T_V2 weights, never trained on KITTI or Sintel) on KITTI 2015 and Sintel
# clean/final, 2 views at native resolution, with the round-1 real-data metrics and the alpha/beta decomposition.
# Any4D is scored from its cached round-1 2-view wide-input predictions (no GPU). Resumable.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/w2_common.sh
w2_setup
( source $XMV/bin/activate; w2_gpu; cd $SRC
  step "raft kitti15"; python e5_w2_realdata.py predict --gt $RD/gt/kitti15 --pass rgb --pred $W2/pred/raft_kitti15_rgb $W2LIM | tail -1
  for ps in clean final; do step "raft sintel $ps"; python e5_w2_realdata.py predict --gt $RD/gt/sintel --pass $ps --pred $W2/pred/raft_sintel_$ps $W2LIM | tail -1; done )
cd $SRC; export OMP_NUM_THREADS=8
step "eval kitti15"
$A4PY e5_w2_realdata.py eval --gt $RD/gt/kitti15 --pass rgb --model raft --pred $W2/pred/raft_kitti15_rgb --out $RESULT_DIR/w2_real_raft_kitti15_rgb.json $W2LIM | tail -1
$A4PY e5_w2_realdata.py eval --gt $RD/gt/kitti15 --pass rgb --model any4d --pred $RD/pred/any4d_kitti15_rgb_2v_wide --out $RESULT_DIR/w2_real_any4d_kitti15_rgb.json $W2LIM | tail -1
for ps in clean final; do
  step "eval sintel $ps"
  $A4PY e5_w2_realdata.py eval --gt $RD/gt/sintel --pass $ps --model raft --pred $W2/pred/raft_sintel_$ps --out $RESULT_DIR/w2_real_raft_sintel_$ps.json $W2LIM | tail -1
  $A4PY e5_w2_realdata.py eval --gt $RD/gt/sintel --pass $ps --model any4d --pred $RD/pred/any4d_sintel_${ps}_2v_wide --out $RESULT_DIR/w2_real_any4d_sintel_$ps.json $W2LIM | tail -1
done
rm -f $RESULT_DIR/*.partial
step "summary"
$A4PY e5_w2_analysis.py realout --real $RESULT_DIR --out $RESULT_DIR | tail -12
ls -la $RESULT_DIR
