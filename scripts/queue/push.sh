#!/bin/bash
# Push one task's results (and the current code) to GitHub. Exit 3 = no repo access yet (supervisor retries).
Q=${E5_QUEUE:-$HOME/e5_multibody/queue}; R=$HOME/e5_multibody/e5-repo; TID=$1
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
mkdir -p results/queue/$TID scripts/queue src
# results: skip big binary arrays and anything over 50 MB
(cd $Q/results/$TID && find . -type f ! -name '*.npy' ! -name '*.npz' -size -50M) | while read f; do
  mkdir -p "results/queue/$TID/$(dirname "$f")"; cp "$Q/results/$TID/$f" "results/queue/$TID/$f"; done
test -f $Q/logs/$TID.log && tail -c 2000000 $Q/logs/$TID.log > results/queue/$TID/task.log
cp $Q/STATUS.md results/queue/STATUS.md 2>/dev/null
cp $Q/supervisor.py $Q/push.sh $Q/enqueue.py $Q/tasks.json $Q/README.md scripts/queue/ 2>/dev/null
cp $Q/tasks/*.sh scripts/queue/ 2>/dev/null
cp $Q/src/*.py src/ 2>/dev/null
git add -A
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git commit -q -m "queue: $TID ($(date '+%Y-%m-%d %H:%M'))"
for i in 1 2 3; do git push -q origin HEAD:$BR && exit 0; sleep 20; git pull -q --rebase -X theirs origin $BR; done
exit 4
