# sourced by every task script
set -o pipefail
# machine settings (A6000: $E5_QUEUE/site_env.sh exports E5_ROOT, E5_BIG, caches, thread caps, GPU headroom)
if [ -z "${E5_SITE_ENV:-}" ] && [ -n "${QDIR:-}" ] && [ -f "$QDIR/site_env.sh" ]; then . "$QDIR/site_env.sh"; fi
E5=${E5_ROOT:-$HOME/e5_multibody}; A4=$E5/Any4D; V2=$E5/data/v2; V1=$E5/data/pilot_v1
BIG=${E5_BIG:-/media/workbench/zone_x/e5_multibody}   # large disk (1.2 TB free): all new data goes here
mkdir -p $BIG/data $BIG/viewsweep $BIG/models $BIG/realdata
ds_dir() { case $1 in v2) echo $V2;; v1|pilot_v1) echo $V1;; *) echo $BIG/data/$1;; esac; }
POST=$E5/.venv-post/bin/python
pick_gpu() {  # waits until some GPU has >= GPU_NEED_GB (default 16) + GPU_HEADROOM_GB (default 0; 4 on the shared A6000) GB free, then prints the freest one
  # GPUs just handed to another of our tasks (reservation files, 300 s, under $QDIR/gpu_resv) count as used, so two tasks starting together do not pick the same GPU
  local need_gb=${GPU_NEED_GB:-16} head_gb=${GPU_HEADROOM_GB:-0} waited=0 rdir=${QDIR:-${TMPDIR:-/tmp}}/gpu_resv g free f p resv eff best bestfree
  local need=$(( (need_gb + head_gb) * 1024 ))
  mkdir -p "$rdir"
  while true; do
    exec 9>"$rdir/.lock"; flock 9
    for f in "$rdir"/*.resv; do [ -e "$f" ] || continue; p=${f##*/}; p=${p#*_}; p=${p%.resv}
      if ! kill -0 "$p" 2>/dev/null || [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 300 ]; then rm -f "$f"; fi; done
    best=""; bestfree=-1
    while IFS=', ' read -r g free; do
      resv=0; for f in "$rdir/${g}"_*.resv; do [ -e "$f" ] && [ "$f" != "$rdir/${g}_$$.resv" ] && resv=$((resv + $(cat "$f"))); done
      eff=$((free - resv)); if [ "$eff" -ge "$need" ] && [ "$eff" -gt "$bestfree" ]; then best=$g; bestfree=$eff; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    if [ -n "$best" ]; then echo $(( need_gb * 1024 )) > "$rdir/${best}_$$.resv"; flock -u 9; exec 9>&-; echo "$best"; return 0; fi
    flock -u 9; exec 9>&-
    [ $((waited % 600)) -eq 0 ] && echo "[pick_gpu] waiting: no GPU has $need MiB free after reservations (need ${need_gb} GB + ${head_gb} GB headroom; waited ${waited}s)" >&2
    sleep 60; waited=$((waited + 60))
  done
}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
any4d_env() { cd $A4 && source .venv/bin/activate && cp $SRC/e5_any4d_infer.py $SRC/e5_any4d_extract.py $SRC/e5_e8_knockout.py $SRC/e5_causal.py $SRC/e5_subspace.py scripts/ 2>/dev/null; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"; }
need() { test -e "$1" || { echo "missing $1"; exit 2; }; }
step() { echo "[$(date '+%H:%M:%S')] $*"; }
