#!/bin/bash
# usage: run_mech_gpu.sh <need_gb> <venv> <log> <python args...>
# Mechanism track (U4/U5). GPU with the most free memory >= need + GPU_HEADROOM_GB (4). One GPU per track: while another
# job of this track is alive (lock dir $E5_BIG/logs/mech/gpu_lock), only that job's GPU is used. Never touches other
# users' processes. CPU threads capped at 12.
NEED=$1; VENV=$2; LOG=$3; shift 3
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
export PYTHONUNBUFFERED=1 E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
mkdir -p $(dirname $LOG)
LOCK=$E5_BIG/logs/mech/gpu_lock; mkdir -p $LOCK
PIN=""
for f in $LOCK/*; do [ -f "$f" ] || continue; p=$(basename $f); if kill -0 $p 2>/dev/null; then PIN=$(cat $f); else rm -f $f; fi; done
[ -n "$MECH_GPU" ] && PIN=$MECH_GPU   # track GPU rule: only this GPU is used
HR=${GPU_HEADROOM_GB:-4}
for i in $(seq 1 ${GPU_WAIT_MIN:-240}); do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d , | awk -v n=$(( (NEED+HR)*1024 )) -v pin="$PIN" '$2>=n && (pin=="" || $1==pin){print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$G" ] && break; sleep 60
done
[ -z "$G" ] && { echo "[run_mech_gpu] no GPU with $((NEED+HR)) GB free (pin=$PIN)" >> $LOG; exit 3; }
echo $G > $LOCK/$$
export CUDA_VISIBLE_DEVICES=$G
echo "[run_mech_gpu] $(date) GPU=$G pin=$PIN need=$NEED venv=$VENV args=$*" >> $LOG
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader >> $LOG
# retry on CUDA OOM (the GPU is shared; outputs are resumable per item): wait for free memory again, then rerun
for att in $(seq 1 ${OOM_RETRIES:-50}); do
  S0=$(wc -c < $LOG)
  taskset -c ${MECH_CPUS:-0-11} $E5_BIG/venvs/$VENV/bin/python "$@" >> $LOG 2>&1
  rc=$?
  [ $rc = 0 ] && break
  tail -c +$((S0+1)) $LOG | grep -q 'OutOfMemoryError' || break
  echo "[run_mech_gpu] OOM (attempt $att), waiting for $((NEED+HR)) GB free on GPU $G $(date)" >> $LOG
  for j in $(seq 1 ${GPU_WAIT_MIN:-240}); do sleep 60
    F=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits); [ "$F" -ge $(( (NEED+HR)*1024 )) ] && break; done
done
rm -f $LOCK/$$
echo "[run_mech_gpu] exit $rc $(date)" >> $LOG
exit $rc
