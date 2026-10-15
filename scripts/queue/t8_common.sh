# shared helpers for the item 8 real-data tasks (sourced after common.sh)
R=$BIG/realdata; mkdir -p $R/gt $R/pred
quiet() { "$@" 2>&1 | { grep -v -i warn || true; }; }      # keeps the exit status of the command
# run_cfg DATASET PASS NAME OFFSETS RESIZE
run_cfg() {
  quiet python scripts/e5_realdata_run.py --src $SRC --gt $R/gt/$1 --pass $2 --offsets $4 --resize $5 \
    --pred $R/pred/any4d_${1}_${2}_${3}_$5 --out $RESULT_DIR/${1}_${2}_${3}_$5.csv ${T8_LIMIT:+--limit $T8_LIMIT}
}
