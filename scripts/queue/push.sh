#!/bin/bash
# Push one task's results (and the current code) to GitHub. Exit 3 = no repo access yet (supervisor retries).
E5R=${E5_ROOT:-$HOME/e5_multibody}; Q=${E5_QUEUE:-$E5R/queue}; R=${E5_PUSH_REPO:-$E5R/e5-repo}; TID=$1  # E5_ROOT default = 4090 path
REMOTE=git@github.com:rohitsinha-iit/e5-multibody-4d-probing.git
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/e5_deploy_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
test -f $HOME/.ssh/e5_deploy_key || { echo "no deploy key"; exit 3; }
if [ ! -d $R/.git ]; then git clone -q $REMOTE $R || exit 3; fi
cd $R || exit 3
git config user.name "e5-queue"; git config user.email "e5-queue@users.noreply.github.com"
git fetch -q origin || exit 3
BR=$(git rev-parse --abbrev-ref HEAD)
# the clone holds no local-only state (every push re-copies from the queue), so sync hard to the remote first
git reset -q --hard origin/$BR
mkdir -p results/queue scripts/queue src
# results of every task (small, human-readable files only), so partial results of running tasks show up too
for RD in $Q/results/*/; do
  T=$(basename $RD); mkdir -p results/queue/$T
  (cd $RD && find . -type f ! -name '*.npy' ! -name '*.npz' -size -50M) | while read f; do
    mkdir -p "results/queue/$T/$(dirname "$f")"; cp "$RD/$f" "results/queue/$T/$f"; done
  test -f $Q/logs/$T.log && tail -c 2000000 $Q/logs/$T.log > results/queue/$T/task.log
done
cp $Q/STATUS.md results/queue/STATUS.md 2>/dev/null
cp $Q/supervisor.py $Q/push.sh $Q/enqueue.py $Q/tasks.json $Q/README.md scripts/queue/ 2>/dev/null
cp $Q/tasks/*.sh scripts/queue/ 2>/dev/null
cp $Q/src/*.py src/ 2>/dev/null
# dashboard (dashboard/README.md renders privately in GitHub; docs/index.html is for GitHub Pages if enabled)
DPY=$(eval echo "$(cat $Q/dashboard_python 2>/dev/null)"); [ -x "$DPY" ] || DPY=${E5_ROOT:-$HOME/e5_multibody}/.venv-post/bin/python
$DPY $Q/src/e5_dashboard.py $Q $R > $Q/logs/dashboard.log 2>&1 || echo "dashboard build failed (see logs/dashboard.log)"
git add -A
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git commit -q -m "queue: $TID ($(date '+%Y-%m-%d %H:%M'))"
for i in 1 2 3; do git push -q origin HEAD:$BR && exit 0; sleep 20; git pull -q --rebase -X theirs origin $BR; done
exit 4
