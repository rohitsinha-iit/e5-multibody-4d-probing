#!/bin/bash
# U2 smoke check: every model on 2 test scenes (and 2 real samples, 2 pose windows) into results/u2_smoke.
set -u
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
export U2_RES=$E5_BIG/results/u2_smoke U2_MAX_ATTEMPTS=1
S=$E5_ROOT/repo/scripts/attr/u2_step.sh; SC=301,303
run() { echo "=== $(date +%T) $*"; bash $S "$@" 2>&1 | grep -v -i 'warn' ; }
run 10 any4d u2_infer.py --model any4d --kubric test:A,test:B,test:C,test:D --scenes $SC --real kitti15:rgb:2v:wide,sintel:final:2v:wide --pose --limit 2
run 16 xm u2_infer.py --model vggt --kubric test:A,test:B,test:C,test:D --scenes $SC --real kitti15:rgb:2v,sintel:final:2v --pose --limit 2
run 15 4rc u2_infer.py --model 4rc --kubric test:A,test:B,test:C,test:D --scenes $SC --real kitti15:rgb:2v,sintel:final:2v --pose --limit 2
run 12 more u2_infer.py --model more --kubric test:B,test:D --scenes $SC --pose --limit 2
run 8 xm u2_infer.py --model cotracker3 --kubric test:C,test:D --scenes $SC --real kitti15:rgb:2v,kitti15:rgb:mv7,sintel:final:2v --limit 2
run 8 xm u2_infer.py --model raft --kubric test:C,test:D --scenes $SC --real kitti15:rgb:2v,sintel:final:2v --limit 2
run 18 opend4rt u2_infer.py --model opend4rt --kubric test:A,test:B,test:C,test:D --scenes $SC --real kitti15:rgb:2v,sintel:final:2v --pose --limit 2
run 12 da3 u2_da3_batch.py --kubric test:C,test:D --scenes $SC --real kitti15:rgb:w12 --limit 2
run 20 trackcraft3r u2_infer.py --model tc3r_gt --kubric test:C,test:D --scenes $SC
run 20 trackcraft3r u2_infer.py --model tc3r_da3 --kubric test:C,test:D --scenes $SC --real kitti15:rgb:w12 --limit 2
run 20 trackcraft3r u2_infer.py --model tc3r_identity --kubric test:D --scenes $SC
echo "=== $(date +%T) scoring"
taskset -c 0-11 $E5_BIG/venvs/post/bin/python -u $E5_ROOT/repo/src/attr/u2_score.py --procs 8
echo "=== SMOKE DONE $(date +%F' '%T)"
