# Round 2, review W5: causal specificity of the five supply heads (2:1, 22:9, 17:10, 14:6, 8:11) on v2 out100.
# 40 matched random 5-head sets (mean ablation), resample ablation from a fixed donor scene (five, each single head,
# 10 random sets), amplification x1.5 (five and 20 random sets). Metrics as round 1 (e5_causal.py).
source $QDIR/tasks/common.sh; set -e
cp $SRC/e5_kubric_post.py $SRC/e5_step3_analysis.py $SRC/e5_multibody.py $V2/
need $V2/out100
any4d_env
python scripts/e5_causal.py --data $V2/out100 --e5 $V2 --out $RESULT_DIR/causal_null_v2.json --protocol null \
  --n_null 40 --n_resample 10 --n_amp 20 --amp 1.5 --donor_seed 0 --limit ${R2_LIMIT:-0} 2>&1 | grep -v -i warn
cd $RESULT_DIR
python $SRC/e5_summarize.py causal_null_v2.json causal_null_v2_conditions slope,epe_static,epe_moving,sep_ratio > /dev/null
python $SRC/e5_causal_null_summary.py causal_null_v2.json $TASK_ID
rm -f causal_null_v2.json.partial
ls -la $RESULT_DIR
