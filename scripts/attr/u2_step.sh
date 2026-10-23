#!/bin/bash
# U2 step runner: waits until GPU 0 has NEED_GB + GPU_HEADROOM_GB free, then runs one u2_infer.py / u2_da3_batch.py
# process pinned to GPU 0 (CUDA_VISIBLE_DEVICES=0) and 12 CPU threads. Retries after OOM or other failures.
# usage: u2_step.sh NEED_GB VENV SCRIPT ARGS...
set -u
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
NEED=$1; VENV=$2; SCRIPT=$3; shift 3
REPO=$E5_ROOT/repo
HEAD=${GPU_HEADROOM_GB:-4}
CORES=${U2_CORES:-0-11}
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in $(seq 1 ${U2_MAX_ATTEMPTS:-30}); do
  while true; do
    FREE=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [ "$FREE" -ge $(( (NEED + HEAD) * 1024 )) ] && break
    echo "[$(date +%F' '%T)] waiting for GPU 0: free ${FREE} MiB < $(( (NEED + HEAD) * 1024 )) MiB"; sleep 120
  done
  echo "[$(date +%F' '%T)] attempt $attempt: $VENV $SCRIPT $*  (GPU 0 free ${FREE} MiB)"
  taskset -c $CORES $E5_BIG/venvs/$VENV/bin/python -u $REPO/src/attr/$SCRIPT "$@"
  rc=$?
  [ $rc -eq 0 ] && exit 0
  echo "[$(date +%F' '%T)] exit code $rc; retry after 300 s"; sleep 300
done
exit $rc
