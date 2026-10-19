#!/bin/bash
# Code-path smoke test of the mechanism track on pilot factorial scenes (test/transfer split: code validation only,
# a handful of components; outputs under $E5_BIG/results/attr_mech/smoke, never used for analysis).
set -u
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
SM=$E5_BIG/results/attr_mech/smoke; mkdir -p $SM
export U4A_RES=$SM/u4a U5_RES=$SM/u5 PROBE_FEAT=$SM/probe_feats
A=$E5_ROOT/repo/src/attr; PY="$E5_BIG/venvs/any4d/bin/python"
S=v2_469
C5=b2h1,b22h9,b8h11,b10r,b23r
st() { echo "=== $1 $(date +%T)"; }
st obj;  $PY $A/interchange.py --mode screen --screen obj --split any --scenes $S --comps $C5 || echo FAIL obj
st cam;  $PY $A/interchange.py --mode screen --screen cam --split any --scenes $S --comps $C5 || echo FAIL cam
st cont; $PY $A/interchange.py --mode screen --screen cont --split any --scenes $S --comps b2h1,b10h3,b22h9,b14h6 || echo FAIL cont
st camwarp; $PY $A/interchange.py --mode screen --screen camwarp --split any --scenes $S --comps b10r,b23r || echo FAIL camwarp
for s in obj cam cont; do $PY $A/interchange.py --mode rank --screen $s --split any --scenes $S || echo FAIL rank $s; done
st joint; $PY $A/interchange.py --mode joint --screen obj --split any --scenes $S --n_null 3 || echo FAIL joint
st joint_cont; $PY $A/interchange.py --mode joint --screen cont --split any --scenes $S --n_null 2 || echo FAIL joint_cont
st necessity; $PY $A/interchange.py --mode necessity --screen obj --split any --scenes v2_469,v2_443 --limit 1 || echo FAIL nec
st probe; $PY $A/motion_probe_allviews.py --mode all --split any --scenes v2_305,v2_401,v2_409,v2_435,v2_443,v2_451 || echo FAIL probe
st probe_eval; $PY $A/motion_probe_allviews.py --mode eval --split transfer --allow_test --scenes v3float_531 --out_tag _smoke || echo FAIL probe_eval
st dirs; $PY $A/repair.py --mode dirs --split any --scenes v2_305,v2_401,v2_409 || echo FAIL dirs
st u5a_i;   $PY $A/repair.py --mode u5a --variant i   --g 2 --split any --scenes $S || echo FAIL u5a_i
st u5a_iii; $PY $A/repair.py --mode u5a --variant iii --g 2 --n_random 2 --split any --scenes $S || echo FAIL u5a_iii
st u5a_iv;  $PY $A/repair.py --mode u5a --variant iv  --g 2 --split any --scenes $S || echo FAIL u5a_iv
st u5a_v;   $PY $A/repair.py --mode u5a --variant v   --split any --scenes $S || echo FAIL u5a_v
st u5b_a;   $PY $A/repair.py --mode u5b --variant a --kstar_cont 2 --split any --scenes $S || echo FAIL u5b_a
st u5b_c;   $PY $A/repair.py --mode u5b --variant c --kstar_cont 2 --n_random 2 --split any --scenes $S || echo FAIL u5b_c
st u5b_d;   $PY $A/repair.py --mode u5b --variant d --split any --scenes $S || echo FAIL u5b_d
st u5b_sintel; $PY $A/repair.py --mode u5b --variant a --kstar_cont 2 --eval sintel --allow_test --limit 1 || echo FAIL u5b_sintel
st u5b_tum; $PY $A/repair.py --mode u5b --variant b --eval tum --allow_test --limit 1 || echo FAIL u5b_tum
st done
