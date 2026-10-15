#!/bin/bash
# Scale E5 v2 to 100 scenes: n3/confound_same/confound_glued 20 each, other splits 10 each (pilot s301-s314 included).
set +e
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -w /work --cpus 8 kubricdockerhub/kubruntu python3 kubric_e5_scene.py"
declare -A EXTRA=( [n2_generic]=8 [n3_generic]=18 [camrot]=8 [epi_aligned]=8 [planar]=8 [confound_same]=18 [confound_glued]=18 )
jobs_list=(); seed=400
for S in n2_generic n3_generic camrot epi_aligned planar confound_same confound_glued; do
  for k in $(seq 1 ${EXTRA[$S]}); do seed=$((seed+1)); jobs_list+=("$S:$seed"); done
done
running=0
for js in "${jobs_list[@]}"; do
  S=${js%%:*}; SEED=${js##*:}
  test -f out/${S}_s${SEED}/e5_meta.json && continue
  $RUN --split $S --seed $SEED --resolution 512x512 --frame_end 24 --samples 32 --min_sep_px 25 --grounded 1 --size_lo 1.4 --size_hi 2.2 --job-dir out/${S}_s${SEED} > logs/${S}_s${SEED}.txt 2>&1 &
  running=$((running+1)); if [ $running -ge 4 ]; then wait -n; running=$((running-1)); fi
done
wait
