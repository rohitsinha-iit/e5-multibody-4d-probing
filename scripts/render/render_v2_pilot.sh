#!/bin/bash
# E5 v2 pilot: confound-reduced scenes (grounded sliding motion, matched object sizes 1.4-2.2 m), 2 per split.
set +e
mkdir -p logs out
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -w /work --cpus 8 kubricdockerhub/kubruntu python3 kubric_e5_scene.py"
# smoke test first (4 frames, 128 px)
$RUN --split confound_glued --seed 1 --resolution 128x128 --frame_end 4 --samples 8 --grounded 1 --size_lo 1.4 --size_hi 2.2 --job-dir smoke_v2 > logs/smoke_v2.txt 2>&1
if ! test -f smoke_v2/e5_meta.json; then echo SMOKE_FAILED; tail -30 logs/smoke_v2.txt; exit 1; fi
rm -rf smoke_v2
jobs_list=(); seed=300
for S in n2_generic n3_generic camrot epi_aligned planar confound_same confound_glued; do
  for k in 0 1; do seed=$((seed+1)); jobs_list+=("$S:$seed"); done
done
running=0
for js in "${jobs_list[@]}"; do
  S=${js%%:*}; SEED=${js##*:}
  test -f out/${S}_s${SEED}/e5_meta.json && continue
  $RUN --split $S --seed $SEED --resolution 512x512 --frame_end 24 --samples 32 --min_sep_px 25 --grounded 1 --size_lo 1.4 --size_hi 2.2 --job-dir out/${S}_s${SEED} > logs/${S}_s${SEED}.txt 2>&1 &
  running=$((running+1)); if [ $running -ge 4 ]; then wait -n; running=$((running-1)); fi
done
wait
