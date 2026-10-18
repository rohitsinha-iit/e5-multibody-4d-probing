# Round 2, review item 1 (W2): one table and one figure for all models (slope, alpha, beta, sep_ratio), the dose-response
# figures, and a rule-based verdict paragraph. CPU only.
source $QDIR/tasks/common.sh; set -e; source $QDIR/tasks/w2_common.sh
KR=${W2_KUBRIC_DIR:-$QDIR/results/w2_kubric_baselines}; RR=${W2_REAL_DIR:-$QDIR/results/w2_real_baselines}; DR=${W2_DOSE_DIR:-$QDIR/results/w2_dose}
$A4PY $SRC/e5_w2_analysis.py summary --kubric $KR --real $RR --dose $DR/w2_dose_stats.json --out $RESULT_DIR | tail -40
cp $DR/fig_w2_dose_*.png $RESULT_DIR/ 2>/dev/null || true
ls -la $RESULT_DIR
