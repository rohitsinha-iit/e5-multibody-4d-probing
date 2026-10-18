#!/bin/bash
# Scale E5 Kubric set to 10 scenes per split (pilot s101-s114 + 8 new per split, seeds 201-256). Same settings as pilot.
set +e
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -w /work --cpus 8 kubricdockerhub/kubruntu python3 kubric_e5_scene.py"
jobs_list=()
seed=200
for S in n2_generic n3_generic camrot epi_aligned planar confound_same confound_glued; do
  for k in 0 1 2 3 4 5 6 7; do seed=$((seed+1)); jobs_list+=("$S:$seed"); done
done
running=0
for js in "${jobs_list[@]}"; do
  S=${js%%:*}; SEED=${js##*:}
  test -f out/${S}_s${SEED}/e5_meta.json && continue
  $RUN --split $S --seed $SEED --resolution 512x512 --frame_end 24 --samples 32 --min_sep_px 25 --job-dir out/${S}_s${SEED} > logs/${S}_s${SEED}.txt 2>&1 &
  running=$((running+1))
  if [ $running -ge 4 ]; then wait -n; running=$((running-1)); fi
done
wait
