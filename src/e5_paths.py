"""Central E5 path configuration.

Every E5 script that needs a machine-specific path should import this module (or, in shell, use
${E5_ROOT:-...} / ${E5_BIG:-...}). The defaults are the original 4090-server paths, so the 4090 keeps
working unchanged; other machines export E5_ROOT and E5_BIG (see scripts/queue/site_env_a6000.sh).

  E5_ROOT   small-disk project root (4090: /home/rohit/e5_multibody). Holds Any4D/, data/v2, data/pilot_v1,
            queue/, .venv-post (on the A6000 some of these are symlinks into E5_BIG).
  E5_BIG    large-disk root (4090: /media/workbench/zone_x/e5_multibody). Holds data/<set>, realdata/, ft/,
            models/{src,hf,venvs}, viewsweep/, ...
  E5_QUEUE  queue directory (default $E5_ROOT/queue).
"""
import os

DEFAULT_E5_ROOT = "/home/rohit/e5_multibody"
DEFAULT_E5_BIG = "/media/workbench/zone_x/e5_multibody"

E5_ROOT = os.path.abspath(os.path.expanduser(os.environ.get("E5_ROOT", DEFAULT_E5_ROOT)))
E5_BIG = os.path.abspath(os.path.expanduser(os.environ.get("E5_BIG", DEFAULT_E5_BIG)))
QUEUE = os.path.abspath(os.path.expanduser(os.environ.get("E5_QUEUE", os.path.join(E5_ROOT, "queue"))))
ANY4D = os.path.join(E5_ROOT, "Any4D")
V2 = os.path.join(E5_ROOT, "data", "v2")
V1 = os.path.join(E5_ROOT, "data", "pilot_v1")
POST_PY = os.path.join(E5_ROOT, ".venv-post", "bin", "python")
TORCH_HOME = os.path.abspath(os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch")))


def big(*parts):
    """Path under E5_BIG."""
    return os.path.join(E5_BIG, *parts)


def root(*parts):
    """Path under E5_ROOT."""
    return os.path.join(E5_ROOT, *parts)


def ds_dir(name):
    """Dataset directory, same rule as ds_dir() in scripts/queue/tasks/common.sh."""
    if name == "v2":
        return V2
    if name in ("v1", "pilot_v1"):
        return V1
    return big("data", name)


def resolve(s):
    """Expand $E5_ROOT / $E5_BIG / ~ placeholders in a string (used for task-definition fields)."""
    env = dict(os.environ, E5_ROOT=E5_ROOT, E5_BIG=E5_BIG, E5_QUEUE=QUEUE)
    out = s
    for k in ("E5_QUEUE", "E5_ROOT", "E5_BIG"):
        out = out.replace("${" + k + "}", env[k]).replace("$" + k, env[k])
    return os.path.expanduser(out)


if __name__ == "__main__":
    for k in ("E5_ROOT", "E5_BIG", "QUEUE", "ANY4D", "V2", "V1", "POST_PY", "TORCH_HOME"):
        print(f"{k}={globals()[k]}")
