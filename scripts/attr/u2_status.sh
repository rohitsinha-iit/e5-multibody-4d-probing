#!/bin/bash
# U2 progress: pipeline status lines, per-unit item counts, item errors, GPU 0 state.
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
U=${U2_RES:-$E5_BIG/results/u2}; L=$E5_BIG/logs/u2
echo "== pipeline status"; tail -n 30 $L/pipeline_status.txt 2>/dev/null
echo "== running"; pgrep -af 'u2_pipeline|u2_infer|u2_da3_batch|u2_score|u2_summary' | grep -v pgrep | cut -c1-160
echo "== items per unit (npz count, DONE marker)"
for d in $(find $U/pred $U/pred_real $U/pred_pose -mindepth 1 -type d ! -name _errors 2>/dev/null | sort); do
  n=$(ls $d 2>/dev/null | grep -c '\.npz$'); [ $n -gt 0 ] || continue
  echo "$n $( [ -f $d/_DONE.json ] && echo DONE || echo .. ) ${d#$U/}"; done
echo "== item errors: $(find $U -path '*_errors*' -name '*.json' 2>/dev/null | wc -l)"
nvidia-smi -i 0 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
