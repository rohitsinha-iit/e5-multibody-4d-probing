
RENDER_IMG=kubricdockerhub/kubruntu
render_one() {  # render_one OUTDIR SPLIT SEED [extra flags...]  (skips finished scenes)
  local D=$1 S=$2 SEED=$3; shift 3
  test -f $D/out/${S}_s${SEED}/e5_meta.json && return 0
  mkdir -p $D/out $D/logs
  docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $D:/work -w /work --cpus 8 $RENDER_IMG python3 kubric_e5_scene.py \
    --split $S --seed $SEED --resolution 512x512 --frame_end 24 --samples 32 --min_sep_px 25 "$@" \
    --job-dir out/${S}_s${SEED} > $D/logs/${S}_s${SEED}.txt 2>&1
  test -f $D/out/${S}_s${SEED}/e5_meta.json || echo "render failed ${S}_s${SEED}"
}
FLAGS_V2="--grounded 1 --size_lo 1.4 --size_hi 2.2"
FLAGS_FLOAT="--grounded 1 --size_lo 1.4 --size_hi 2.2 --float_height 0.6 --no_shadows 1"
# scene lists: "dataset split seed"; the pilot is the subset marked P
scene_list() {
  local s
  for s in 501 502 503; do echo "v3far confound_farglued $s P"; done
  for s in $(seq 504 520); do echo "v3far confound_farglued $s -"; done
  echo "v3float epi_aligned 531 P"; echo "v3float epi_aligned 532 P"; for s in $(seq 533 560); do echo "v3float epi_aligned $s -"; done
  echo "v3float n2_generic 561 P"; for s in $(seq 562 570); do echo "v3float n2_generic $s -"; done
  echo "v3float confound_same 571 P"; for s in $(seq 572 580); do echo "v3float confound_same $s -"; done
  echo "v3float n3_generic 581 P"; for s in $(seq 582 590); do echo "v3float n3_generic $s -"; done
  for s in $(seq 591 600); do echo "v3float confound_glued $s -"; done
  for s in $(seq 701 720); do echo "v2 epi_aligned $s -"; done
}
render_list() {  # render_list P|ALL : 4 renders in parallel
  local running=0 ds sp seed tag D F
  while read ds sp seed tag; do
    [ "$1" = "P" ] && [ "$tag" != "P" ] && continue
    D=$(ds_dir $ds); mkdir -p $D; cp $SRC/kubric_e5_scene.py $D/
    if [ $ds = v3float ]; then F=$FLAGS_FLOAT; else F=$FLAGS_V2; fi
    render_one $D $sp $seed $F &
    running=$((running+1)); if [ $running -ge 4 ]; then wait -n; running=$((running-1)); fi
  done < <(scene_list)
  wait
}
post_ds() {  # post_ds DATASET_DIR : flow check, SNR, contact sheets, GIFs
  local D=$1; cp $SRC/e5_post_driver.py $SRC/e5_kubric_post.py $SRC/e5_multibody.py $D/
  (cd $D && $POST e5_post_driver.py out > post.log 2>&1; tail -3 post.log)
}
