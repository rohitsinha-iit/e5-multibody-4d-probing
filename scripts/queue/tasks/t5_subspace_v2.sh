# Item 5: LEACE erasure of the linear motion subspace at layers 17-21, and steering along it (v2).
source $QDIR/tasks/common.sh; set -e
cp $SRC/e5_kubric_post.py $SRC/e5_step3_analysis.py $SRC/e5_multibody.py $V2/
any4d_env
python scripts/e5_subspace.py --data $V2/out --gt $V2/gt_patches --internals $V2/internals --e5 $V2 \
  --out $RESULT_DIR/subspace_v2.json 2>&1 | grep -v -i warn
cd $RESULT_DIR && python $SRC/e5_summarize.py subspace_v2.json subspace_v2 slope,epe_moving,epe_static_obj,epe_bg,spurious_static_obj
