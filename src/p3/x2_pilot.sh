#!/bin/bash
# P3 X2 / U3 pilot pipeline on the A6000 (one GPU). Usage: bash src/p3/x2_pilot.sh [pilot_steps]
set -u
export E5_ROOT=/data2/DATA/rohit_sinha/e5 E5_BIG=/data2/DATA/rohit_sinha/e5/big
C=$E5_ROOT/cache; export PIP_CACHE_DIR=$C/pip UV_CACHE_DIR=$C/uv XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch HF_HOME=$E5_BIG/models/hf TMPDIR=$C/tmp
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 HF_HUB_OFFLINE=1
PY=$E5_BIG/venvs/p3-wan/bin/python; LOG=$E5_BIG/logs/cause; STEPS=${1:-8000}
cd $E5_ROOT/repo
pick_gpu() {  # $1 = GB needed (+4 GB margin); prints the GPU with most free memory, waits until one fits
  local need=$(( ($1 + 4) * 1024 ))
  while true; do
    local best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
    local idx=${best%%,*}; local free=${best##*, }
    if [ "$free" -ge "$need" ]; then echo $idx; return; fi
    sleep 60
  done
}
stage() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pilot_pipeline.log; }
until grep -aq SETUP_DONE $LOG/setup.log; do sleep 60; done
stage "setup done"
if [ ! -f $E5_BIG/models/ckpt/wan_empty_ctx.pt ]; then
  G=$(pick_gpu 14); stage "ctx on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY src/p3/wan_feats.py ctx >> $LOG/ctx.log 2>&1; stage "ctx exit $?"
fi
if [ ! -f $E5_BIG/results/p3_g0a.json ] && [ "${SKIP_G0A:-0}" != 1 ]; then
  G=$(pick_gpu 24); stage "g0a on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY src/p3/wan_feats.py g0a >> $LOG/g0a.log 2>&1; stage "g0a exit $?"
fi
# features: extract repeatedly until the clip build for shards 0010-0059 has finished
for k in $(seq 1 100); do
  G=$(pick_gpu 16); stage "extract pass $k on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY src/p3/wan_feats.py extract >> $LOG/extract.log 2>&1 || { stage "extract failed"; exit 1; }
  grep -aq "CLIPS2_EXIT" $LOG/clips_build.log && { CUDA_VISIBLE_DEVICES=$G $PY src/p3/wan_feats.py extract >> $LOG/extract.log 2>&1; break; }
  sleep 300
done
NF=$(ls $E5_BIG/p3/feat/ | grep -c "^[0-9].*[0-9].npy$"); stage "features done: $NF clips"
[ "$NF" -ge 100 ] || { stage "too few features, stopping"; exit 1; }
$PY src/p3/p3_train.py --name bench --param abs --loss L3D --bench_loader 20 --workers 8 --preload 0 >> $LOG/loader_bench.log 2>&1
$PY src/p3/p3_train.py --name bench --param abs --loss L3D --bench_loader 100 --workers 8 --preload 1 --tracks_keep 16384 >> $LOG/loader_bench.log 2>&1
stage "loader bench done"
G=$(pick_gpu 16); stage "pilot P-abs L3D ($STEPS steps) on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY src/p3/p3_train.py --name pilot_abs_L3D_s0 --param abs --loss L3D --seed 0 --steps $STEPS \
   --warmup 1000 --ckpt_every 1000 --eval_every 500 --workers 8 --preload 1 --tracks_keep 16384 >> $LOG/pilot_train.log 2>&1
stage "pilot exit $?"
echo PIPELINE_DONE >> $LOG/pilot_pipeline.log
