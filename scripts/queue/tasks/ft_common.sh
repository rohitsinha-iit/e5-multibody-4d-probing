# item 10 (fine-tuning) helpers, sourced after common.sh
FT=${FT_ROOT:-$BIG/ft}; mkdir -p $FT/runs $FT/eval
FT_V2=$BIG/data/ft_train_v2; FT_FLOAT=$BIG/data/ft_train_float
FT_DATA=${FT_DATA_OVERRIDE:-"$FT_V2/out,$FT_FLOAT/out"}
ft_ckpt() { if [ "$1" = base ]; then echo $A4/checkpoints/any4d_4v_combined.pth; else echo $FT/runs/$1/any4d_ft.pth; fi; }
ft_eval_data() { case $1 in v2) echo $V2/out100;; v3float) echo $BIG/data/v3float/out;; v3far) echo $BIG/data/v3far/out;; esac; }
# training scene list: "split seed" ; 200 per generator, seeds 2001-2200 (v2 grounded) and 2201-2400 (floating)
ft_scene_list() {  # ft_scene_list v2|float
  local s0 sp n i; [ "$1" = v2 ] && s0=2001 || s0=2201
  for sp_n in epi_aligned:40 n2_generic:30 n3_generic:30 camrot:25 planar:25 confound_same:25 confound_glued:25; do
    sp=${sp_n%%:*}; n=${sp_n##*:}
    for i in $(seq 1 $n); do echo "$sp $s0"; s0=$((s0+1)); done
  done
}
ft_render() {  # ft_render v2|float : 4 renders in parallel, resumable; FT_RENDER_LIMIT=N renders only N scenes
  local G=$1 D F running=0 k=0 sp seed
  if [ $G = v2 ]; then D=$FT_V2; F=$FLAGS_V2; else D=$FT_FLOAT; F=$FLAGS_FLOAT; fi
  mkdir -p $D; cp $SRC/kubric_e5_scene.py $D/
  while read sp seed; do
    k=$((k+1)); [ -n "$FT_RENDER_LIMIT" ] && [ $k -gt $FT_RENDER_LIMIT ] && break
    render_one $D $sp $seed $F &
    running=$((running+1)); if [ $running -ge 4 ]; then wait -n || true; running=$((running-1)); fi
  done < <(ft_scene_list $G)
  wait || true
}
