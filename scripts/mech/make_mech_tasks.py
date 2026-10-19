#!/usr/bin/env python3
"""Generate E5 queue tasks for the mechanism track (U4a, U4b gate, U5a, U5b; plan section 9 names).
Writes scripts/queue/tasks/<id>.sh and scripts/queue/task_defs/attr_mech_fit.json (fit split: allowed now) and
attr_mech_test.json (test/transfer/real: gate 'attr_prereg', opened by touching $E5_QUEUE/approvals/attr_prereg after
the Oct 27 pre-registration). Every task picks one GPU through common.sh pick_gpu (GPU_NEED_GB + 4 GB headroom).
"""
import json, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TD = os.path.join(REPO, "scripts", "queue", "tasks"); DD = os.path.join(REPO, "scripts", "queue", "task_defs")
HDR = """#!/bin/bash
# {title}
source $QDIR/tasks/common.sh; set -e
export GPU_NEED_GB={need}; export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "GPU $CUDA_VISIBLE_DEVICES"
export E5_REPO=$E5_ROOT/repo ANY4D_DIR=$E5_ROOT/Any4D
A=$E5_ROOT/repo/src/attr; PY=$E5_BIG/venvs/{venv}/bin/python
"""
tasks = {"fit": [], "test": []}


def add(split, tid, title, body, deps=(), need=14, venv="any4d", gate=None):
    open(os.path.join(TD, tid + ".sh"), "w").write(HDR.format(title=title, need=need, venv=venv) + body + "\n")
    os.chmod(os.path.join(TD, tid + ".sh"), 0o755)
    tasks[split].append(dict(id=tid, title=title, script=tid + ".sh", deps=list(deps), gate=gate, max_attempts=3,
                             min_free_gb=20, disk_path="$E5_BIG"))


NCH = 4
# ---------------- U4a fit split (allowed now)
scr = []
for s in ("obj", "cam", "cont", "objall"):
    for c in range(NCH):
        tid = f"u4a_screen_{s}_{c}"
        add("fit", tid, f"U4a {s} screen on fit scenes, component chunk {c}/{NCH}",
            f"step {tid}; $PY $A/interchange.py --mode screen --screen {s} --split fit --chunk {c}/{NCH}")
        scr.append(tid)
add("fit", "u4a_screen_camwarp_0", "U4a secondary warp-aligned camera screen (24 residual states) on fit scenes",
    "$PY $A/interchange.py --mode screen --screen camwarp --split fit --comps " + ",".join(f"b{b}r" for b in range(24)))
add("fit", "u4a_rank", "U4a rank components on fit scenes (obj, cam, cont, camwarp)",
    "for s in obj cam cont camwarp objall; do $PY $A/interchange.py --mode rank --screen $s --split fit; done", deps=scr + ["u4a_screen_camwarp_0"], need=0)
add("fit", "u4a_joint_fit", "U4a joint top-k patches on fit scenes (completeness, no nulls) and k*",
    "for s in obj cam cont; do $PY $A/interchange.py --mode joint --screen $s --split fit --n_null 0; "
    "$PY $A/interchange.py --mode kstar --screen $s --split fit; done", deps=["u4a_rank"])
add("fit", "u4a_necessity_fit", "U4a resample-ablation necessity of top components on fit scenes",
    "for s in obj cam; do $PY $A/interchange.py --mode necessity --screen $s --split fit; done", deps=["u4a_rank"])
add("fit", "u5_probe_any4d", "U5 all-view motion probe on Any4D (fit scenes; L* by CV; 90% recall threshold)",
    "$PY $A/motion_probe_allviews.py --mode all --split fit")
add("fit", "u5a_dirs", "U5a counterfactual directions (P_obj at k* + five E5 heads) on fit scenes",
    "K=$($PY -c \"import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['obj']['kstar'] or 0)\"); "
    "$PY $A/repair.py --mode dirs --split fit --kstar_obj $K", deps=["u4a_joint_fit"])
add("fit", "u4b_gate_g0h", "Gate G0h: VGGT camera-token layout (identity and camera-head positive check)",
    "$PY $A/vggt_hooks.py", venv="xm", need=12)
# ---------------- after pre-registration (gate attr_prereg)
G = "attr_prereg"
KC = "KC=$($PY -c \"import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['cont']['kstar'] or 5)\"); "
KO = "KO=$($PY -c \"import json;print(json.load(open('$E5_BIG/results/u4a/kstar.json'))['obj']['kstar'] or 0)\"); "
for s in ("obj", "cam", "cont"):
    add("test", f"u4a_joint_{s}", f"U4a joint top-k with 40 matched nulls on test scenes ({s})",
        f"$PY $A/interchange.py --mode joint --screen {s} --split test --allow_test --n_null 40", deps=["u4a_joint_fit"], gate=G)
    add("test", f"u4a_null_{s}_transfer", f"U4a joint top-k with 40 nulls on v3float transfer ({s})",
        f"$PY $A/interchange.py --mode joint --screen {s} --split transfer --allow_test --n_null 40", deps=["u4a_joint_fit"], gate=G)
for v in ("i", "ii", "iii", "iv"):
    for g in ("1.5", "2"):
        add("test", f"u5a_edit_{v}_g{g}", f"U5a edit variant {v}, g={g}, test and transfer",
            KO + f"for sp in test transfer; do $PY $A/repair.py --mode u5a --variant {v} --g {g} --kstar_obj $KO --split $sp --allow_test; done",
            deps=["u5a_dirs"], gate=G)
add("test", "u5a_edit_v", "U5a variant v: heads_sel LoRA, test and transfer",
    "for sp in test transfer; do $PY $A/repair.py --mode u5a --variant v --split $sp --allow_test; done", deps=[], gate=G, need=16)
for v in ("a", "b", "c", "d"):
    add("test", f"u5b_mask_{v}_kubric", f"U5b masking variant {v} on Kubric test scenes",
        KC + f"$PY $A/repair.py --mode u5b --variant {v} --kstar_cont $KC --split test --allow_test",
        deps=["u5_probe_any4d", "u4a_joint_fit"], gate=G)
    if v != "d":
        for st in ("sintel", "tum", "bonn"):
            add("test", f"u5b_mask_{v}_{st}", f"U5b masking variant {v} on {st}",
                KC + f"$PY $A/repair.py --mode u5b --variant {v} --kstar_cont $KC --eval {st} --allow_test",
                deps=["u5_probe_any4d", "u4a_joint_fit"], gate=G)
add("test", "u5_probe_eval_any4d", "U5 probe AUC and mask IoU on test and transfer",
    "for sp in test transfer; do $PY $A/motion_probe_allviews.py --mode eval --split $sp --allow_test; done", deps=["u5_probe_any4d"], gate=G)
for k, v in tasks.items():
    json.dump(v, open(os.path.join(DD, f"attr_mech_{k}.json"), "w"), indent=1)
    print(k, len(v), "tasks ->", os.path.join(DD, f"attr_mech_{k}.json"))
