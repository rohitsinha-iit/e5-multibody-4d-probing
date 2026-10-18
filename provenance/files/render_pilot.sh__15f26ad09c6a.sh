#!/bin/bash
# Pilot E5 Kubric set: 2 scenes per split, 512 px, 24 frames. Runs 4 renders at a time (8 CPUs each).
set +e
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -w /work --cpus 8 kubricdockerhub/kubruntu python3 kubric_e5_scene.py"
jobs_list=()
seed=100
for S in n2_generic n3_generic camrot epi_aligned planar confound_same confound_glued; do
  for k in 0 1; do seed=$((seed+1)); jobs_list+=("$S:$seed"); done
done
running=0
for js in "${jobs_list[@]}"; do
  S=${js%%:*}; SEED=${js##*:}
  $RUN --split $S --seed $SEED --resolution 512x512 --frame_end 24 --samples 32 --min_sep_px 25 --job-dir out/${S}_s${SEED} > logs/${S}_s${SEED}.txt 2>&1 &
  running=$((running+1))
  if [ $running -ge 4 ]; then wait -n; running=$((running-1)); fi
done
wait
