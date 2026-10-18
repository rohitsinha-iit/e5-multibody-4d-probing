# Moving E5 / P3 / Idea 1 work from the 4090 server to the A6000 server

Written 2026-10-18 for the period when the 4090 server (10.0.117.206) is unavailable. It goes with docs/Unified_E5_P3_Idea1_implementation_plan.md.

## 1. What is in this repo and what is not

The repo holds all code, all task definitions, generator versions and environment specs:

| Path | Contents |
|---|---|
| src/ | every file in the queue's src (analysis, models, Kubric generator, real-data fetch) |
| src/any4d_side/ | E5 scripts that lived inside the Any4D clone |
| scripts/queue/ | supervisor, push, enqueue, all task scripts (tasks/), task definitions (task_defs/, tasks/defs/) and a snapshot of the queue state |
| provenance/ | every older copy of a script that produced a dataset or result, deduplicated by sha256, with MANIFEST.tsv mapping original server path to file |
| provenance/kubric_e5_scene_USED_FOR_v2_v3float_v3far.py | the exact generator for the v2, v3float and v3far scenes (sha256 f65cfd0b...). The current src/kubric_e5_scene.py is newer (c6e4d25a...). The MotionFactorial renders (U1) must start from the provenance version, so that variant D reproduces the existing scenes. |
| env/ | lockfiles of the three uv venvs (venv_any4d.txt, venv_xm.txt, venv_post.txt), pinned commits of external repos (external_repos.tsv: Any4D, vggt, map-anything, PAGE4D, co-tracker, dinov2 hub), the Kubric docker image id, Hugging Face cache revisions, crontab, GPU list, and hardcoded_paths.txt |
| data/MANIFEST_4090.tsv | every dataset on the 4090 server with size and file count. No data is in git. |

Not in the repo: data, model weights, venvs, logs, and the deploy key.

## 2. A6000 server facts that change the setup (from the host notes)
- Home is /data1/rohit_sinha. /data1 was nearly full (about 55 GB free on 2026-10-08). Check `df -h /data1 /data2` first.
- Admin rule: datasets go to ~/DATA, a symlink to the shared /data2/DATA. Pass cache_dir or root for Hugging Face downloads so they land there.
- conda is installed and GitHub is reachable. Whether docker is available (needed for Kubric renders) is not known yet: check with `docker info`.
- No scheduler. Other users run long jobs, including Wan servers on GPUs 2 and 3. Check nvidia-smi before every run and pin with CUDA_VISIBLE_DEVICES.
- call_command has a 60 s limit, so long setup steps run as jobs.

## 3. Bootstrap order
1. `git clone https://github.com/rohitsinha-iit/e5-multibody-4d-probing ~/e5` (public repo; no key needed for reading).
2. Choose roots. Code: ~/e5. Data and weights: /data2/DATA/rohit_sinha/e5 (later called $E5_BIG).
3. Paths. The code has hardcoded 4090 paths (/home/rohit/e5_multibody, /media/workbench/zone_x/e5_multibody); see env/hardcoded_paths.txt for files and line counts. /media cannot be symlinked without root. The first engineering task on the A6000 is therefore to read two environment variables, E5_ROOT (default the old home path) and E5_BIG (default the old /media path), in a small src/e5_paths.py, and replace the literals. The defaults keep the 4090 server working unchanged.
4. Environments. Recreate the three venvs from env/venv_*.txt (uv or conda python 3.12). The torch builds in the lockfiles are cu124, which works with the A6000 driver 550. Clone the external repos at the commits in env/external_repos.tsv and install them editable. Pre-clone dinov2 into ~/.cache/torch/hub/facebookresearch_dinov2_main, to avoid the torch.hub GitHub rate limit seen on the 4090.
5. Weights. Re-download from Hugging Face at the revisions in env/models_and_checkpoints.txt (Any4D airlabshare/any4d-checkpoint, VGGT-1B, VGGT_tracker_fixed, MapAnything rev 6f3a25bf, PAGE-4D, CoTracker3), into $E5_BIG/models/hf.
6. Queue. Copy scripts/queue/* to $E5_BIG/queue (or ~/e5/queue), set E5_QUEUE, and add the cron lines from env/crontab.txt with the new paths. Pushing results from the A6000 needs a new deploy key, or a token-based push started by the agent. The 4090 deploy key must not be copied.

## 4. Data needed for the unified plan, and how to move it

| Dataset (4090 path) | Size | Needed for |
|---|---|---|
| ~/e5_multibody/data/v2 (out100 subset used) | 23 GB total | U1 base scenes D, U2, U4, U5 |
| $BIG/data/v3float | 11 GB | U1, U2 transfer split |
| $BIG/data/v4epi | 6.9 GB | U2 rows |
| $BIG/realdata/kitti15, sintel, gt | 12.6 GB | U2 real data (these can also be re-fetched with src/e5_realdata_fetch.sh) |
| $BIG/ft/runs (five-head LoRA checkpoints) | see manifest | U5a baseline (v) |
| $BIG/data/ft_train_v2, ft_train_float | 28 GB | only to retrain the E5 LoRA; optional |

Decision (user, 2026-10-18): only the irreplaceable files were uploaded. Everything else is downloaded or regenerated on the A6000.

- Hugging Face, private dataset rohitsinhaiitgn/e5-multibody-data: ft_runs_heads_ctrl.tar.part000/001 (five-head LoRA heads_sel and the random-head control ctrl_sel, merged Any4D checkpoints, 5.43 GB), SHA256SUMS, README.md. Sizes were verified after upload. Restore:

      huggingface-cli download rohitsinhaiitgn/e5-multibody-data --repo-type dataset --local-dir <dst>
      cd <dst> && sha256sum -c SHA256SUMS && cat ft_runs_heads_ctrl.tar.part* | tar -xf - -C $E5_BIG/ft/runs

- Re-render on the A6000 (needs docker for the Kubric image): v2, v3float and v4epi from provenance/kubric_e5_scene_USED_FOR_v2_v3float_v3far.py with the original seeds. These form variant D of MotionFactorial, together with the new A, B and C variants. The re-renders may not be bit-identical to the 4090 renders, so E5 numbers that are reused in the paper must be re-scored on the new D renders.
- Re-download on the A6000: KITTI 2015 and Sintel (src/e5_realdata_fetch.sh), pose sets (yslan/pointmap_regression_evalsets), CoTracker3_Kubric shards, and all model weights at the revisions in env/models_and_checkpoints.txt.

A full 63.8 GB tar staging copy of all datasets remains on the 4090 server at /media/workbench/zone_x/e5_multibody/hf_stage/parts. It can be deleted once the A6000 re-renders are verified.
