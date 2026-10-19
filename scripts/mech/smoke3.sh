#!/bin/bash
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
SM=$E5_BIG/results/attr_mech/smoke; export U4A_RES=$SM/u4a U5_RES=$SM/u5 PROBE_FEAT=$SM/probe_feats
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/any4d/bin/python
rm -rf $SM/u4a/joint_obj; echo "=== joint $(date +%T)"; $PY $A/interchange.py --mode joint --screen obj --split any --scenes v2_469 --n_null 2 || echo FAIL joint
$PY $A/interchange.py --mode kstar --screen obj --split any --scenes v2_469 || echo FAIL kstar
echo "=== done"
