#!/bin/bash
# Resumable download + extraction of the public real-data sets for item 8 (no tokens, public URLs only).
# Usage: bash e5_realdata_fetch.sh [kitti] [kitti_mv] [sintel]   (default: all)
# Layout: $R/kitti15/{training/...}, $R/sintel/{training/...}. A marker file .done_<part> records completion,
# archives are deleted after extraction, partial downloads are resumed with curl -C -.
set -o pipefail
BIG=/media/workbench/zone_x/e5_multibody; R=$BIG/realdata; mkdir -p $R/dl
S3=https://s3.eu-central-1.amazonaws.com/avg-kitti
SIN=https://files.is.tue.mpg.de
get() {  # get URL DEST PART [unzip patterns...]
  local url=$1 dest=$2 part=$3; shift 3
  [ -f $dest/.done_$part ] && { echo "have $part"; return 0; }
  mkdir -p $dest; local z=$R/dl/$(basename $url)
  local n=$(curl -sIL -m 30 $url | grep -i '^content-length' | tail -1 | awk '{print $2}' | tr -d '\r')
  local tries=0
  while [ "$(stat -c%s $z 2>/dev/null || echo 0)" != "$n" ]; do
    tries=$((tries+1)); [ $tries -gt 8 ] && { echo "download failed $url"; return 1; }
    echo "[$(date +%T)] downloading $url ($n bytes, have $(stat -c%s $z 2>/dev/null || echo 0))"
    curl -sS -L -C - --retry 5 --retry-delay 10 -o $z $url || sleep 20
  done
  echo "[$(date +%T)] extracting $part"
  unzip -q -o $z "$@" -d $dest || { echo "unzip failed $z"; return 1; }
  touch $dest/.done_$part && rm -f $z && echo "[$(date +%T)] done $part"
}
parts=${@:-kitti kitti_mv sintel}
rc=0
for p in $parts; do case $p in
  kitti) get $S3/data_scene_flow.zip $R/kitti15 sf 'training/*' || rc=1
         get $S3/data_scene_flow_calib.zip $R/kitti15 calib 'training/*' || rc=1;;
  kitti_mv) get $S3/data_scene_flow_multiview.zip $R/kitti15 mv 'training/image_2/*' || rc=1;;
  sintel) get $SIN/sintel/MPI-Sintel-complete.zip $R/sintel complete 'training/*' || rc=1
          get $SIN/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip $R/sintel depth 'training/depth/*' 'training/camdata_left/*' || rc=1;;
esac; done
du -sh $R/* 2>/dev/null
exit $rc
