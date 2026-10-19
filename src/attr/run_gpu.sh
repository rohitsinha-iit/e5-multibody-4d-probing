#!/bin/bash
# usage: run_gpu.sh <need_gb> <venv> <log> <python args...>
# Picks the GPU with the most free memory >= need+4 GB. One GPU per track: while another run_gpu.sh job of this track
# is alive (lock dir $E5_BIG/logs/newmodels/gpu_lock), only that job's GPU is used (waits for memory there).
NEED=$1; VENV=$2; LOG=$3; shift 3
export E5_BIG=/data2/DATA/rohit_sinha/e5/big
export XDG_CACHE_HOME=/data2/DATA/rohit_sinha/e5/cache HF_HOME=$E5_BIG/models/hf TORCH_HOME=$E5_BIG/models/torch PIP_CACHE_DIR=/data2/DATA/rohit_sinha/e5/cache/pip
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTHONUNBUFFERED=1
LOCK=$E5_BIG/logs/newmodels/gpu_lock; mkdir -p $LOCK
PIN=""
for f in $LOCK/*; do [ -f "$f" ] || continue; p=$(basename $f); if kill -0 $p 2>/dev/null; then PIN=$(cat $f); else rm -f $f; fi; done
for i in $(seq 1 180); do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d , | awk -v n=$(( (NEED+4)*1024 )) -v pin="$PIN" '$2>=n && (pin=="" || $1==pin){print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$G" ] && break; sleep 60
done
[ -z "$G" ] && { echo "no GPU with $((NEED+4)) GB free (pin=$PIN)" >> $LOG; exit 3; }
echo $G > $LOCK/$$
export CUDA_VISIBLE_DEVICES=$G
echo "[run_gpu] $(date) GPU=$G pin=$PIN need=$NEED venv=$VENV args=$*" >> $LOG
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader >> $LOG
/usr/bin/time -v $E5_BIG/venvs/$VENV/bin/python "$@" >> $LOG 2>&1
rc=$?
rm -f $LOCK/$$
echo "[run_gpu] exit $rc $(date)" >> $LOG
exit $rc
