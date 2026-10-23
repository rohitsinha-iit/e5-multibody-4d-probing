#!/bin/bash
# Step 1 of the mechanism test launch: fit-derived values on FIT scenes only (v2 even seeds), GPU 1 only.
#  a. joint top-k (k in 1,3,5,10,20, no nulls) for objall, cam, cont, obj -> k* per screen (results/u4a/kstar.json)
#  b. necessity (resample ablation, donor = random other fit scene) for the top-k heads of every screen
#  c. all-view motion probe: extract, CV over L in 13..21, 90%-recall threshold (results/u5b/probe_any4d.*)
#  d. U5a projectors for P_obj (objall top-k*) union the five E5 heads (results/u5a/u5a_dirs*.pt)
#  e. freeze: results/u4a/fit_derived_values.json (U5b list, random head sets, sha256 of every file)
# Resumable: every step skips finished per-scene outputs. Log: $E5_BIG/logs/mech/fit_pipeline.log
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
export MECH_GPU=${MECH_GPU:-1} GPU_WAIT_MIN=${GPU_WAIT_MIN:-2880}
R=$E5_ROOT/repo; A=$R/src/attr; L=$E5_BIG/logs/mech; RES=$E5_BIG/results
RUN="$R/scripts/mech/run_mech_gpu.sh 14 any4d"
log() { echo "[fit_pipeline] $(date '+%F %T') $*" >> $L/fit_pipeline.log; }
log start
for s in objall cam cont obj; do
  log "joint fit $s"; $RUN $L/fit_joint_$s.log $A/interchange.py --mode joint --screen $s --split fit --n_null 0 || log "joint $s rc $?"
  $E5_BIG/venvs/any4d/bin/python $A/interchange.py --mode kstar --screen $s --split fit >> $L/fit_kstar.log 2>&1
done
for s in objall cam cont obj; do
  log "necessity fit $s"; $RUN $L/fit_necessity_$s.log $A/interchange.py --mode necessity --screen $s --split fit || log "necessity $s rc $?"
done
log "probe"; $RUN $L/fit_probe.log $A/motion_probe_allviews.py --mode all --split fit || log "probe rc $?"
POBJ=$($E5_BIG/venvs/any4d/bin/python -c "import json;k=json.load(open('$RES/u4a/kstar.json'))['objall'];print(','.join(k['topk_heads'] or []))")
log "dirs P_obj=$POBJ"; $RUN $L/fit_dirs.log $A/repair.py --mode dirs --split fit --p_obj "$POBJ" || log "dirs rc $?"
log "freeze"; $E5_BIG/venvs/any4d/bin/python $A/freeze_fit.py >> $L/fit_pipeline.log 2>&1
log done
