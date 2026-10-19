#!/bin/bash
# Mechanism-track launcher. usage: launch_mech.sh fit | test | watch
#   fit   enqueue the fit-split tasks (U4a screens, rank, joint/k*, necessity, probe, U5a directions, G0h)
#   test  enqueue the post-pre-registration tasks; they wait on gate attr_prereg until
#         touch $E5_QUEUE/approvals/attr_prereg   (lead, after freezing results/attr_preregistration.json on Oct 27)
#   watch run the fit-split U4a screens directly (no supervisor) on fit scenes as their renders land (fit_watch.sh)
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
R=$E5_ROOT/repo
python3 $R/scripts/mech/make_mech_tasks.py
case "$1" in
  fit)  python3 $E5_QUEUE/enqueue.py $R/scripts/queue/task_defs/attr_mech_fit.json ;;
  test) python3 $E5_QUEUE/enqueue.py $R/scripts/queue/task_defs/attr_mech_test.json ;;
  watch) nohup bash $R/scripts/mech/fit_watch.sh > $E5_BIG/logs/mech/fit_watch.out 2>&1 & echo "watch pid $!" ;;
  *) echo "usage: $0 fit|test|watch"; exit 1 ;;
esac
