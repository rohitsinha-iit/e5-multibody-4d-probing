#!/bin/bash
# U2 new-model scene gates, sequential on one GPU (run_gpu.sh pins the track's GPU). Scenes: first 10 complete v2 scenes.
D=/data2/DATA/rohit_sinha/e5/big/data/v2/out; O=/data2/DATA/rohit_sinha/e5/big/results/newmodels; L=/data2/DATA/rohit_sinha/e5/big/logs/newmodels; RG=/data2/DATA/rohit_sinha/e5/repo/src/attr/run_gpu.sh; GN=/data2/DATA/rohit_sinha/e5/repo/src/attr/gate_newmodels.py
nok(){ n=0; for s in $D/*; do [ -f $s/gt.npz ] && [ -f $s/e5_meta.json ] && n=$((n+1)); done; echo $n; }
for i in $(seq 1 36); do [ $(nok) -ge 10 ] && break; echo "waiting for 10 scenes ($(nok)) $(date)"; sleep 300; done
echo "start $(date); complete scenes: $(nok)"
$RG 14 4rc $L/gate_4rc.log $GN --model 4rc --data $D --n 10 --out $O/gate_4rc.json
$RG 18 opend4rt $L/gate_opend4rt.log $GN --model opend4rt --data $D --n 10 --out $O/gate_opend4rt.json
$RG 10 more $L/gate_more.log $GN --model more --data $D --n 10 --out $O/gate_more.json
$RG 22 trackcraft3r $L/gate_trackcraft3r.log $GN --model trackcraft3r --cond gt --data $D --n 10 --out $O/gate_trackcraft3r.json
$RG 12 any4d $L/gate_any4d.log $GN --model any4d --data $D --n 10 --out $O/gate_any4d.json
/data2/DATA/rohit_sinha/e5/big/venvs/tools/bin/python -c "import numpy" 2>/dev/null || /data2/DATA/rohit_sinha/e5/big/venvs/tools/bin/pip install -q numpy
/data2/DATA/rohit_sinha/e5/big/venvs/tools/bin/python /data2/DATA/rohit_sinha/e5/repo/src/attr/gate_newmodels_compare.py $O /data2/DATA/rohit_sinha/e5/big/results/attr_gates.json
echo "end $(date)"
