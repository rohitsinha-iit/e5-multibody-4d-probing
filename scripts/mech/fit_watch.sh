#!/bin/bash
# Fit-split U4a screens run directly (no supervisor) as the Data track's fit renders land.
# Every 20 min: list complete fit scenes (v2 even seeds with A,B,C,D and gt_factorial) and run the obj, cam, cont and
# camwarp (24 residual states) screens on those without results (interchange.py is resumable per scene), then rank.
# One GPU at a time through run_mech_gpu.sh (lock $E5_BIG/logs/mech/gpu_lock). Never touches test scenes.
# Stops when 50 fit scenes are screened, at 2026-10-27 23:59 IST, or when $E5_BIG/logs/mech/STOP_WATCH exists.
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
R=$E5_ROOT/repo; A=$R/src/attr; L=$E5_BIG/logs/mech; mkdir -p $L
PY=$E5_BIG/venvs/any4d/bin/python
WARP=$(for b in $(seq 0 23); do echo -n "b${b}r,"; done); WARP=${WARP%,}
END=$(date -d '2026-10-27 23:59' +%s)
while true; do
  [ -e $L/STOP_WATCH ] && { echo "[watch] STOP_WATCH $(date)"; break; }
  [ $(date +%s) -gt $END ] && { echo "[watch] past Oct 27 $(date)"; break; }
  N=$(cd $A && $PY -c "import interchange as ic; print(len(ic.split_scenes('fit')))" 2>/dev/null)
  TODO=$(cd $A && $PY -c "
import interchange as ic, os, glob
n=0
for d in ic.split_scenes('fit'):
    for s in ('obj','cam','cont','camwarp','objall'):
        if not glob.glob(os.path.join(ic.RES, 'screen_'+s, ic.sid(d)+'*.json')): n+=1
print(n)" 2>/dev/null)
  echo "[watch] $(date '+%F %T') fit scenes complete: $N, screens to do: $TODO"
  if [ "${TODO:-0}" -gt 0 ]; then
    for s in obj cam cont objall; do $R/scripts/mech/run_mech_gpu.sh 14 any4d $L/fit_screen_$s.log $A/interchange.py --mode screen --screen $s --split fit; done
    $R/scripts/mech/run_mech_gpu.sh 14 any4d $L/fit_screen_camwarp.log $A/interchange.py --mode screen --screen camwarp --split fit --comps $WARP
    for s in obj cam cont camwarp objall; do (cd $A && $PY interchange.py --mode rank --screen $s --split fit >> $L/fit_rank.log 2>&1); done
  fi
  DONE=$(cd $A && $PY -c "
import interchange as ic, os, glob
print(sum(all(glob.glob(os.path.join(ic.RES,'screen_'+s,ic.sid(d)+'*.json')) for s in ('obj','cam','cont','camwarp')) for d in ic.split_scenes('fit')))" 2>/dev/null)
  [ "${DONE:-0}" -ge 50 ] && { echo "[watch] 50 fit scenes screened $(date)"; break; }
  sleep 1200
done
