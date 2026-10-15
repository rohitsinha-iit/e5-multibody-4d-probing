# Full E5 pipeline (Steps 3-6 + adjacency control) on dataset $1 with stored layers $2 (default: all 25).
source $QDIR/tasks/common.sh; set -e
DS=$1; LAY=${2:-}; TAG=${3:-$DS}; D=$(ds_dir $DS); need $D/out
cp $SRC/e5_step3_analysis.py $SRC/e5_gt_patches.py $SRC/e5_internal_analysis.py $SRC/e5_adjacency_control.py $SRC/e5_multibody.py $SRC/e5_kubric_post.py $D/
any4d_env
step "$DS inference"; python scripts/e5_any4d_infer.py --data $D/out --out $D/pred_any4d_7v 2>&1 | grep -v -i warn | tail -2
step "$DS extraction"
if [ -n "$LAY" ]; then python scripts/e5_any4d_extract.py --data $D/out --out $D/internals --layers $LAY 2>&1 | grep -v -i warn | tail -2
else python scripts/e5_any4d_extract.py --data $D/out --out $D/internals 2>&1 | grep -v -i warn | tail -2; fi
cd $D
R=$RESULT_DIR/$TAG; mkdir -p $R
step "$DS step3"; python e5_step3_analysis.py out pred_any4d_7v $R/step3_any4d_7v.json > step3.log 2>&1
step "$DS gt patches"; python e5_gt_patches.py out pred_any4d_7v gt_patches > gt_patches.log 2>&1
ALL=$(python -c "import json,os;p='internals/stored_layers.json';print(','.join(map(str,json.load(open(p)) if os.path.exists(p) else range(25))))")
L1=$(python -c "a='$ALL'.split(',');print(','.join(a[:len(a)//2+1]))"); L2=$(python -c "a='$ALL'.split(',');print(','.join(a[len(a)//2+1:]))")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
step "$DS internal analysis (A, F in background; P on GPU)"
test -f $R/partA_results.json || (E5_PARTS=A E5_LAYERS=$ALL python e5_internal_analysis.py internals gt_patches $R/partA > partA.log 2>&1) & a1=$!
test -f $R/partF1_results.json || (E5_PARTS=F E5_LAYERS=$L1 python e5_internal_analysis.py internals gt_patches $R/partF1 > partF1.log 2>&1) & a2=$!
test -f $R/partF2_results.json || (E5_PARTS=F E5_LAYERS=$L2 python e5_internal_analysis.py internals gt_patches $R/partF2 > partF2.log 2>&1) & a3=$!
test -f $R/partP1_results.json || E5_PARTS=P E5_LAYERS=$L1 python e5_internal_analysis.py internals gt_patches $R/partP1 > partP1.log 2>&1
test -f $R/partP2_results.json || E5_PARTS=P E5_LAYERS=$L2 python e5_internal_analysis.py internals gt_patches $R/partP2 > partP2.log 2>&1
OMP_NUM_THREADS=8 python e5_adjacency_control.py internals gt_patches $R/adjacency_control.json > adj.log 2>&1
wait $a1; wait $a2; wait $a3
cp scene_summary.json $R/ 2>/dev/null || true
ls -la $R
