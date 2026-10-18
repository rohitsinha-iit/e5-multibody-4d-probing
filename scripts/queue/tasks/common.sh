# sourced by every task script
set -o pipefail
E5=$HOME/e5_multibody; A4=$E5/Any4D; V2=$E5/data/v2; V1=$E5/data/pilot_v1
BIG=/media/workbench/zone_x/e5_multibody   # large disk (1.2 TB free): all new data goes here
mkdir -p $BIG/data $BIG/viewsweep $BIG/models $BIG/realdata
ds_dir() { case $1 in v2) echo $V2;; v1|pilot_v1) echo $V1;; *) echo $BIG/data/$1;; esac; }
POST=$E5/.venv-post/bin/python
pick_gpu() {  # waits until some GPU has >= ${GPU_NEED_GB:-16} GB free (other users share these GPUs), then prints the freest one
  local need=$(( ${GPU_NEED_GB:-16} * 1024 )) waited=0 line
  while true; do
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
    if [ "${line#*, }" -ge "$need" ]; then echo "${line%%,*}"; return 0; fi
    [ $((waited % 600)) -eq 0 ] && echo "[pick_gpu] waiting: best GPU has ${line#*, } MiB free, need $need MiB (waited ${waited}s)" >&2
    sleep 60; waited=$((waited + 60))
  done
}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
any4d_env() { cd $A4 && source .venv/bin/activate && cp $SRC/e5_any4d_infer.py $SRC/e5_any4d_extract.py $SRC/e5_e8_knockout.py $SRC/e5_causal.py $SRC/e5_subspace.py scripts/ 2>/dev/null; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"; }
need() { test -e "$1" || { echo "missing $1"; exit 2; }; }
step() { echo "[$(date '+%H:%M:%S')] $*"; }
