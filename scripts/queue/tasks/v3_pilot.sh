# Item 1+2 pilot: far-glued rigid pairs (C3) and floating shadowless size-matched scenes (C4), for visual approval.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh
set -e
D=$BIG/data/smoke_v3; mkdir -p $D; cp $SRC/kubric_e5_scene.py $D/
step "smoke renders"
for S in confound_farglued epi_aligned; do
  F=$FLAGS_V2; [ $S = epi_aligned ] && F=$FLAGS_FLOAT
  docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $D:/work -w /work --cpus 8 $RENDER_IMG python3 kubric_e5_scene.py \
    --split $S --seed 1 --resolution 128x128 --frame_end 4 --samples 8 $F --job-dir s_$S > $D/smoke_$S.txt 2>&1 || true
  test -f $D/s_$S/e5_meta.json || { echo "SMOKE FAILED $S"; tail -40 $D/smoke_$S.txt; exit 1; }
  grep -i "shadows disabled" $D/smoke_$S.txt || true
done
rm -rf $D
step "pilot renders"
render_list P
for ds in v3far v3float; do
  post_ds $(ds_dir $ds)
  mkdir -p $RESULT_DIR/$ds; cp $(ds_dir $ds)/sheets/* $RESULT_DIR/$ds/ 2>/dev/null; cp $(ds_dir $ds)/scene_summary.json $RESULT_DIR/$ds/
done
ls $RESULT_DIR/*/ | head -40
