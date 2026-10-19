#!/bin/bash
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
SM=$E5_BIG/results/attr_mech/smoke; export U4A_RES=$SM/u4a U5_RES=$SM/u5 PROBE_FEAT=$SM/probe_feats
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
echo "=== objall $(date +%T)"; $PY $A/interchange.py --mode screen --screen objall --split any --scenes v2_469 --comps b2h1,b22h9,b10r,b21r || echo FAIL objall
echo "=== joint $(date +%T)"; $PY $A/interchange.py --mode joint --screen obj --split any --scenes v2_469 --n_null 3 || echo FAIL joint
echo "=== joint_cont $(date +%T)"; $PY $A/interchange.py --mode joint --screen cont --split any --scenes v2_469 --n_null 2 || echo FAIL joint_cont
echo "=== kstar $(date +%T)"; $PY $A/interchange.py --mode kstar --screen obj --split any --scenes v2_469 || echo FAIL kstar
echo "=== g0h $(date +%T)"; cd $A && $E5_BIG/venvs/xm/bin/python $A/vggt_hooks.py || echo FAIL g0h
echo "=== done $(date +%T)"
