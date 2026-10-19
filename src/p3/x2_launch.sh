#!/bin/bash
# P3 X2 / U3 main runs: {P-abs, P-res} x {L3D, L2D} x seeds {0, 1} (8 runs) + capacity cell (P-abs, L3D, width 1024).
# DO NOT RUN before results/p3_preregistration.json exists (thresholds frozen by the lead on Oct 27).
# One GPU at a time for this track; runs are sequential; each run is resumable (rerun the script to continue).
# G2 rule (P3 v2): every cell must reach static EPE within 10 % of the best cell on held-out windows; a lagging cell may
# train up to 1.5x steps (STEPS_LAG); record it in results/p3_x2_g2_extensions.json.
# Usage: bash src/p3/x2_launch.sh [STEPS=20000]
set -u
export E5_ROOT=/data2/DATA/rohit_sinha/e5 E5_BIG=/data2/DATA/rohit_sinha/e5/big
C=$E5_ROOT/cache; export XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch HF_HOME=$E5_BIG/models/hf TMPDIR=$C/tmp HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
PY=$E5_BIG/venvs/p3-wan/bin/python; LOG=$E5_BIG/logs/cause; STEPS=${1:-20000}
cd $E5_ROOT/repo
[ -f $E5_BIG/results/p3_preregistration.json ] || { echo "no pre-registration file: refusing to start"; exit 1; }
pick_gpu() { local need=$(( ($1 + 4) * 1024 )); while true; do
  local best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
  [ "${best##*, }" -ge "$need" ] && { echo ${best%%,*}; return; }; sleep 60; done; }
COMMON="--steps $STEPS --warmup 1000 --ckpt_every 1000 --eval_every 1000 --workers 8 --preload 1 --tracks_keep 16384"
for seed in 0 1; do for param in abs res; do for loss in L3D L2D; do
  name=x2_${param}_${loss}_s${seed}
  grep -q '"step": '$STEPS $E5_BIG/p3/runs/$name/final_eval.json 2>/dev/null && continue
  G=$(pick_gpu 20); echo "[$(date '+%F %T')] $name on GPU $G" >> $LOG/x2_main.log
  CUDA_VISIBLE_DEVICES=$G $PY src/p3/p3_train.py --name $name --param $param --loss $loss --seed $seed $COMMON >> $LOG/$name.log 2>&1
done; done; done
name=x2_cap_abs_L3D_w1024_s0
G=$(pick_gpu 28); echo "[$(date '+%F %T')] $name on GPU $G" >> $LOG/x2_main.log
CUDA_VISIBLE_DEVICES=$G $PY src/p3/p3_train.py --name $name --param abs --loss L3D --seed 0 --width 1024 $COMMON >> $LOG/$name.log 2>&1
# evaluation on E5 splits and MotionFactorial C and D (when the Data track has produced them)
RUNS=$(ls -d $E5_BIG/p3/runs/x2_* | paste -sd,)
G=$(pick_gpu 20)
for split in v2 v3float; do
  [ -d $E5_BIG/data/$split ] && CUDA_VISIBLE_DEVICES=$G $PY src/p3/x2_eval.py --runs $RUNS --data $E5_BIG/data/$split --tag $split --variant D >> $LOG/x2_eval.log 2>&1
done
for split in v2 v3float; do for v in C D; do
  d=$E5_BIG/attr/factorial/$split/_variant_$v   # placeholder: point at the Data track's per-variant scene list (see report)
  [ -d $d ] && CUDA_VISIBLE_DEVICES=$G $PY src/p3/x2_eval.py --runs $RUNS --data $d --tag fact_${split}_$v --variant $v --si_ref $E5_BIG/data/$split >> $LOG/x2_eval.log 2>&1
done; done
echo X2_MAIN_DONE >> $LOG/x2_main.log
