# Item 4: five supply heads together, amplification, path patching into the scene-flow decoder (v2, 100 scenes).
source $QDIR/tasks/common.sh; set -e
cp $SRC/e5_kubric_post.py $SRC/e5_step3_analysis.py $SRC/e5_multibody.py $V2/
any4d_env
python scripts/e5_causal.py --data $V2/out --e5 $V2 --out $RESULT_DIR/causal_v2.json 2>&1 | grep -v -i warn
cd $RESULT_DIR && python $SRC/e5_summarize.py causal_v2.json causal_v2 slope,epe_static,epe_moving,sep_ratio
