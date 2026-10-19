#!/bin/bash
# P3 X2 / U3: complete the training set (CoTracker3_Kubric shards 0000-0399 -> about 2400 clips), extract fp32 Wan
# features, then run the full-data calibration pilot (P-abs, L3D, 20k steps). Resumable: rerun to continue.
set -u
export E5_ROOT=/data2/DATA/rohit_sinha/e5 E5_BIG=/data2/DATA/rohit_sinha/e5/big
C=$E5_ROOT/cache; export XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch HF_HOME=$E5_BIG/models/hf TMPDIR=$C/tmp
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
PY=$E5_BIG/venvs/p3-wan/bin/python; LOG=$E5_BIG/logs/cause; LAST=${1:-399}
D=$E5_BIG/data/cotracker3_kubric; mkdir -p $D/_tgz
stage() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/full_data.log; }
echo "cause track (U3) is downloading shards 0060-$LAST here with <id>.done markers; skip shards that have a .done file; started $(date)" > $D/_DOWNLOADING_BY_CAUSE_TRACK.txt
stage "download start"
for n in $(seq 60 $LAST); do i=$(printf %04d $n)
  [ -f $D/$i.done ] && continue
  /data1/rohit_sinha/miniconda3/bin/python3 -c "from huggingface_hub import hf_hub_download as h; h('facebook/CoTracker3_Kubric','$i.tar.gz',repo_type='dataset',local_dir='$D/_tgz')" >> $LOG/shards_dl.log 2>&1 && \
  tar -xzf $D/_tgz/$i.tar.gz -C $D && rm -f $D/_tgz/$i.tar.gz && touch $D/$i.done && echo ok $i $(date +%T) >> $LOG/shards_dl.log
done
rm -f $D/_DOWNLOADING_BY_CAUSE_TRACK.txt
stage "download done: $(ls $D/*.done | wc -l) shards"
cd $E5_ROOT/repo
OMP_NUM_THREADS=8 $PY src/p3/kubric_shards.py >> $LOG/clips_build.log 2>&1; stage "clips: $(ls $E5_BIG/p3/clips | wc -l)"
export HF_HUB_OFFLINE=1
pick_gpu() { local need=$(( ($1 + 4) * 1024 )); while true; do
  local best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
  [ "${best##*, }" -ge "$need" ] && { echo ${best%%,*}; return; }; sleep 60; done; }
G=$(pick_gpu 12); stage "extract fp32 on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY src/p3/wan_feats.py extract --dit_dtype fp32 >> $LOG/extract_full.log 2>&1 || { stage "extract failed"; exit 1; }
stage "features: $(ls $E5_BIG/p3/feat | grep -c '[0-9].npy$')"
# RAM guard: the preload needs about 26 MB per clip (fp16 features + JPEG RGB + 8192 tracks); wait for that + 25 GB headroom
NEED=$(( $(ls $E5_BIG/p3/feat | grep -c '[0-9].npy$') * 26 / 1024 + 25 ))
until [ $(free -g | awk '/Mem:/{print $7}') -ge $NEED ]; do stage "waiting for ${NEED} GB available RAM"; sleep 600; done
G=$(pick_gpu 12); stage "pilot2 P-abs L3D full data 20k on GPU $G (RAM need ${NEED} GB)"
CUDA_VISIBLE_DEVICES=$G $PY src/p3/p3_train.py --name pilot2_abs_L3D_s0_full --param abs --loss L3D --seed 0 --steps 20000 \
   --warmup 1000 --ckpt_every 1000 --eval_every 1000 --workers 12 --preload 1 --tracks_keep 8192 --rgb_jpeg 95 >> $LOG/pilot2_train.log 2>&1
stage "pilot2 exit $?"
CUDA_VISIBLE_DEVICES=$G $PY src/p3/x2_pilot_diag.py $E5_BIG/p3/runs/pilot2_abs_L3D_s0_full >> $LOG/pilot2_diag.log 2>&1
stage "FULL_DONE"
