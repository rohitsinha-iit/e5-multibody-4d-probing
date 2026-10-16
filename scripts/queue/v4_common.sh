# v4 epipolar set (review item W3). Sourced after common.sh and common_extra.sh.
# epi_accel and epi_const share seeds 801-830 (paired: same scene, frames 0 and 23 identical); epi_sizerand 831-860;
# epi_sizematch 861-890. Pilot = first two seeds of each block. All with v3float settings (floating, shadowless, size-matched).
V4DS=v4epi
v4_scene_list() {
  local s t
  for s in $(seq 801 830); do t=-; [ $s -le 802 ] && t=P; echo "epi_accel $s $t"; echo "epi_const $s $t"; done
  for s in $(seq 831 860); do t=-; [ $s -le 832 ] && t=P; echo "epi_sizerand $s $t"; done
  for s in $(seq 861 890); do t=-; [ $s -le 862 ] && t=P; echo "epi_sizematch $s $t"; done
}
v4_render_list() {  # v4_render_list P|ALL : 4 renders in parallel, skips finished scenes
  local D=$(ds_dir $V4DS) running=0 sp seed tag
  mkdir -p $D; cp $SRC/kubric_e5_scene.py $D/
  while read sp seed tag; do
    [ "$1" = "P" ] && [ "$tag" != "P" ] && continue
    render_one $D $sp $seed $FLAGS_FLOAT &
    running=$((running+1)); if [ $running -ge 4 ]; then wait -n || true; running=$((running-1)); fi
  done < <(v4_scene_list)
  wait || true
  echo "v4 scenes rendered: $(ls $D/out 2>/dev/null | wc -l)"
}
v4_smoke_renders() {  # 128 px, 4 frames, 8 spp, one seed per new split; fails loudly
  local D=$BIG/data/smoke_v4 S; rm -rf $D; mkdir -p $D; cp $SRC/kubric_e5_scene.py $D/
  for S in epi_accel epi_const epi_sizerand epi_sizematch; do
    docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $D:/work -w /work --cpus 8 $RENDER_IMG python3 kubric_e5_scene.py \
      --split $S --seed 801 --resolution 128x128 --frame_end 4 --samples 8 $FLAGS_FLOAT --job-dir s_$S > $D/smoke_$S.txt 2>&1 || true
    test -f $D/s_$S/e5_meta.json || { echo "SMOKE FAILED $S"; tail -40 $D/smoke_$S.txt; return 1; }
  done
  rm -rf $D
}
