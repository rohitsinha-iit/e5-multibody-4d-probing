#!/bin/bash
# Step 2 of the mechanism test launch (U4a, U5a, U5b on test, transfer and real data), GPU 1 only, one resumable
# pipeline. Refuses to start unless the fit_derived_values block is in the pre-registration. Every stage skips
# finished per-item outputs, so re-running the script resumes. Ends with the summary (src/attr/u4u5_summary.py).
# Smoke: SCENES=v2_301,v2_303 TAG=_smoke LIMIT=2 U5A_RES=... U5B_RES=... bash test_pipeline.sh
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
export MECH_GPU=${MECH_GPU:-1} GPU_WAIT_MIN=${GPU_WAIT_MIN:-2880}
R=$E5_ROOT/repo; A=$R/src/attr; L=$E5_BIG/logs/mech; RES=$E5_BIG/results; PY=$E5_BIG/venvs/any4d/bin/python
TT=${TAG:-}; SC=${SCENES:+--scenes $SCENES}; LM=${LIMIT:+--limit $LIMIT}
LOGP=$L/test_pipeline$TT.log; FZ=$RES/u4a/fit_derived_values.json; WIN=$RES/u2/pose_windows.json
RUN="$R/scripts/mech/run_mech_gpu.sh 14 any4d"
log() { echo "[test_pipeline] $(date '+%F %T') $*" >> $LOGP; }
grep -q '"frozen_at"' $RES/attr_preregistration.json || { log "fit_derived_values not in the pre-registration: refusing"; exit 2; }
[ -f $FZ ] || { log "missing $FZ"; exit 2; }
K() { $PY -c "import json;print(json.load(open('$FZ'))['k_used']['$1'])"; }
WARP=$(for b in $(seq 0 23); do echo -n "b${b}r,"; done); WARP=${WARP%,}
joint() { local s=$1 sp=$2 k; k=$(K $s); [ -z "$k" ] && { log "k* is None for $s: joint skipped"; return; }
  log "joint $s $sp k*=$k"; $RUN $L/test_joint_${s}_${sp}$TT.log $A/interchange.py --mode joint --screen $s --split $sp \
    --allow_test --frozen $FZ --n_null 40 --null_ks $k --kgrid 1,3,5,10,20 --out_tag _${sp}$TT $SC || log "joint $s $sp rc $?"; }
u5a() { log "u5a $1 $2 $3"; $RUN $L/test_u5a_${1}_${2}_${3}$TT.log $A/repair.py --mode u5a --variant $1 --eval $2 --split $3 \
    --allow_test $SC $LM || log "u5a $1 $2 $3 rc $?"; }
u5b_real() { log "u5b real"; $RUN $L/test_u5b_real$TT.log $A/repair.py --mode u5b --eval real --allow_test --windows $WIN $LM \
    || log "u5b real rc $?"; }
log "start TAG=$TT SCENES=$SCENES LIMIT=$LIMIT"
joint objall test; joint cam test
u5a all kubric test
log "u5b kubric test"; $RUN $L/test_u5b_kubric$TT.log $A/repair.py --mode u5b --eval kubric --split test --allow_test $SC || log "u5b kubric rc $?"
REAL_DONE=0; [ -f $WIN ] && { u5b_real; REAL_DONE=1; } || log "pose_windows.json not yet written: U5b real deferred"
joint cont test; joint obj test
for s in objall cam cont obj; do log "necessity $s"; $RUN $L/test_necessity_$s$TT.log $A/interchange.py --mode necessity \
    --screen $s --split test --allow_test --out_tag _test$TT $SC || log "necessity $s rc $?"; done
u5a v kubric test
for s in objall cam cont obj; do joint $s transfer; done
u5a all kubric transfer; u5a v kubric transfer
for ds in kitti15 sintel; do u5a all $ds test; u5a v $ds test; done
for sp in test transfer; do log "camwarp $sp"; $RUN $L/test_camwarp_$sp$TT.log $A/interchange.py --mode screen --screen camwarp \
    --split $sp --allow_test --comps $WARP --out_tag _$sp$TT $SC || log "camwarp rc $?"; done
for sp in test transfer; do log "probe eval $sp"; $RUN $L/test_probe_eval$TT.log $A/motion_probe_allviews.py --mode eval --split $sp \
    --allow_test --out_tag "$TT" $SC || log "probe eval rc $?"; done
log "objall screen test (H3 on test)"; $RUN $L/test_screen_objall$TT.log $A/interchange.py --mode screen --screen objall --split test \
    --allow_test --out_tag _test$TT $SC || log "objall screen rc $?"
if [ $REAL_DONE = 0 ]; then
  for i in $(seq 1 $(( ${WAIT_WINDOWS_H:-72} * 6 ))); do [ -f $WIN ] && break; sleep 600; done
  [ -f $WIN ] && u5b_real || log "pose_windows.json still missing after ${WAIT_WINDOWS_H:-72} h: U5b real not run"
fi
[ -z "$TT" ] && { log summary; $PY $A/u4u5_summary.py >> $LOGP 2>&1; }
log done
