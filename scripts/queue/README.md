# E5 experiment queue

Sequential, crash-safe task runner on the GPU server.

- `supervisor.py` runs `tasks.json` in order, one task at a time. Cron starts it every 5 minutes and at boot; a lock
  keeps a single instance alive. On restart, a task that was running is re-queued; task scripts skip work already on disk.
- After each task finishes, `push.sh` commits `results/<task>/`, the task log, the task scripts and `src/` to the
  GitHub repo (deploy key `~/.ssh/e5_deploy_key`, write access to this repo only). Failed pushes retry every 10 minutes.
- Gates: tasks with `"gate": "<name>"` wait until `approvals/<name>` exists (used for visual approval of new datasets).
- `touch approvals/retry_<task>` re-queues a failed task; `touch STOP` makes the supervisor exit after the current task.
- `STATUS.md` and `events.log` show progress.
