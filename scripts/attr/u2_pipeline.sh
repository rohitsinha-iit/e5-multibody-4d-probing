#!/bin/bash
# U2 behaviour pipeline (unified plan U2; settings frozen in results/attr_preregistration.json). Single resumable chain:
# one process per model (model loaded once), each over its units (split:variant, real dataset:pass:config, pose
# windows); every scene / sample / window is skipped when its npz exists, and every unit gets a _DONE.json.
# GPU 0 only (u2_step.sh waits for free memory and retries failures), 12 CPU threads.
# After every model step the CPU scoring (incremental) and the summary are refreshed, so partial tables exist early.
# Re-run safely at any time:  bash scripts/attr/u2_pipeline.sh  (log: $E5_BIG/logs/u2/pipeline.log)
set -u
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
REPO=$E5_ROOT/repo; S=$REPO/scripts/attr/u2_step.sh; L=$E5_BIG/logs/u2; mkdir -p $L $E5_BIG/results/u2
POST=$E5_BIG/venvs/post/bin/python
ALL4="test:A,test:B,test:C,test:D,transfer:A,transfer:B,transfer:C,transfer:D,fit:A,fit:B,fit:C,fit:D"
CD="test:C,test:D,transfer:C,transfer:D,fit:C,fit:D"
BD="test:B,test:D,transfer:B,transfer:D,fit:B,fit:D"
REAL3="kitti15:rgb:2v,sintel:final:2v,sintel:clean:2v"
status() { echo "$(date +%F' '%T) $*" | tee -a $L/pipeline_status.txt; }
refresh() {
  taskset -c 0-11 $POST -u $REPO/src/attr/u2_score.py --procs 10 >> $L/score.log 2>&1
  taskset -c 0-11 $POST -u $REPO/src/attr/u2_summary.py >> $L/summary.log 2>&1
}
step() {  # step NAME NEED_GB VENV SCRIPT ARGS...
  local n=$1; shift
  status "start $n"
  bash $S "$@" > $L/$n.log 2>&1; local rc=$?
  status "end $n rc=$rc"
  refresh
}
status "pipeline start (pid $$)"
$POST -c "import sys; sys.path.append('$REPO/src/attr'); import u2_common as u; w=u.make_pose_windows(); print('pose windows', w['n_windows'], w['per_set'])" >> $L/pipeline_status.txt 2>&1
step any4d      10 any4d        u2_infer.py --model any4d --kubric $ALL4 --real kitti15:rgb:2v:wide,kitti15:rgb:2v:square,sintel:final:2v:wide,sintel:clean:2v:wide,sintel:final:2v:square,sintel:clean:2v:square --pose
step vggt       16 xm           u2_infer.py --model vggt --kubric $ALL4 --real $REAL3 --pose
step 4rc        15 4rc          u2_infer.py --model 4rc --kubric $ALL4 --real $REAL3 --pose
step more       12 more         u2_infer.py --model more --kubric $BD --pose
step cotracker3  8 xm           u2_infer.py --model cotracker3 --kubric $CD --real kitti15:rgb:2v,kitti15:rgb:mv7
step raft        8 xm           u2_infer.py --model raft --kubric $CD --real $REAL3
step opend4rt   18 opend4rt     u2_infer.py --model opend4rt --kubric $ALL4 --real $REAL3 --pose
step da3        12 da3          u2_da3_batch.py --kubric $CD --real kitti15:rgb:w12
step tc3r_gt    20 trackcraft3r u2_infer.py --model tc3r_gt --kubric $CD
step tc3r_da3   20 trackcraft3r u2_infer.py --model tc3r_da3 --kubric $CD --real kitti15:rgb:w12
step tc3r_identity 20 trackcraft3r u2_infer.py --model tc3r_identity --kubric test:D
# longest real-data runs last (CoTracker3 queries every evaluation pixel of Sintel, about 0.4 M per sample)
step cotracker3_sintel 8 xm     u2_infer.py --model cotracker3 --real sintel:final:2v,sintel:clean:2v,sintel:final:mv4,sintel:clean:mv4
# completeness check over all units
$POST - >> $L/pipeline_status.txt 2>&1 <<'EOF'
import glob, os, json
U = os.environ.get("U2_RES", "/data2/DATA/rohit_sinha/e5/big/results/u2")
done = sorted(glob.glob(U + "/pred*/**/_DONE.json", recursive=True)); errs = sorted(glob.glob(U + "/pred*/**/_errors/*.json", recursive=True))
print("units done", len(done), "item errors", len(errs))
json.dump(dict(units_done=[d[len(U) + 1:] for d in done], item_errors=[e[len(U) + 1:] for e in errs]), open(U + "/pipeline_units.json", "w"), indent=1)
EOF
status "pipeline end"
