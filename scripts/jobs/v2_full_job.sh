set +e
export PATH=$HOME/.local/bin:$PATH; W=$PWD; D=${E5_ROOT:-$HOME/e5_multibody}/data/v2
cp render_v2_scale.sh e5_step3_analysis.py e5_gt_patches.py e5_internal_analysis.py e5_adjacency_control.py e5_multibody.py e5_kubric_post.py $D/
cd $D && bash render_v2_scale.sh
echo "scenes $(ls out | wc -l) meta $(ls out/*/e5_meta.json | wc -l)"; grep -l Traceback logs/*.txt || true
${E5_ROOT:-$HOME/e5_multibody}/.venv-post/bin/python e5_post_driver.py out > post.log 2>&1; tail -2 post.log
cp scene_summary.json $W/scene_summary_v2.json
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
cd ${E5_ROOT:-$HOME/e5_multibody}/Any4D && source .venv/bin/activate && cp $W/e5_any4d_infer.py $W/e5_any4d_extract.py scripts/
GPU=$(pick_gpu); echo "GPU $GPU"; nvidia-smi --query-gpu=index,memory.used --format=csv
test "$(ls $D/pred_any4d_7v/*.npz 2>/dev/null | wc -l)" -ge 100 || CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_any4d_infer.py --data $D/out --out $D/pred_any4d_7v --frames 0,4,8,12,16,20,23 > $D/infer.log 2>&1; echo "infer rc $?"
test "$(ls $D/internals/*_att.npz 2>/dev/null | wc -l)" -ge 100 || CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_any4d_extract.py --data $D/out --out $D/internals > $D/extract.log 2>&1; echo "extract rc $?"
cd $D
python e5_step3_analysis.py out pred_any4d_7v step3_any4d_7v.json > step3.log 2>&1; echo "step3 rc $?"
python e5_gt_patches.py out pred_any4d_7v gt_patches > gt_patches.log 2>&1; echo "gt rc $?"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
L1=$(seq -s, 0 12); L2=$(seq -s, 13 24)
E5_PARTS=A python e5_internal_analysis.py internals gt_patches partA > partA.log 2>&1 & a1=$!
E5_PARTS=F E5_LAYERS=$L1 python e5_internal_analysis.py internals gt_patches partF1 > partF1.log 2>&1 & a2=$!
E5_PARTS=F E5_LAYERS=$L2 python e5_internal_analysis.py internals gt_patches partF2 > partF2.log 2>&1 & a3=$!
GPU=$(pick_gpu); echo "probe GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU E5_PARTS=P E5_LAYERS=$L1 python e5_internal_analysis.py internals gt_patches partP1 > partP1.log 2>&1; echo "P1 rc $?"
CUDA_VISIBLE_DEVICES=$GPU E5_PARTS=P E5_LAYERS=$L2 python e5_internal_analysis.py internals gt_patches partP2 > partP2.log 2>&1; echo "P2 rc $?"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU python e5_adjacency_control.py internals gt_patches adjacency_control.json > adj.log 2>&1; echo "adj rc $?"
wait $a1; echo "A rc $?"; wait $a2; echo "F1 rc $?"; wait $a3; echo "F2 rc $?"
cp step3_any4d_7v.json part*_results.json adjacency_control.json *.log $W/ 2>/dev/null; ls -lh $W/*.json
