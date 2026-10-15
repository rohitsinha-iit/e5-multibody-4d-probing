"""Build the E5 queue dashboard inside the GitHub repo clone.

Usage: python e5_dashboard.py <queue_dir> <repo_dir>
Writes
  <repo>/dashboard/README.md            status table + per-run sections (renders privately in the GitHub UI)
  <repo>/dashboard/assets/<task>/*.png  plots made from each task's results, plus figures the task saved itself
  <repo>/docs/index.html                the same content as a static page (for GitHub Pages, only if the user enables it)
Never fails the push: every plot is wrapped, and errors are listed at the bottom of the page.
"""
import glob, html, json, os, shutil, sys, time, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q, R = sys.argv[1], sys.argv[2]
REPO_URL = "https://github.com/rohitsinha-iit/e5-multibody-4d-probing"
ASSETS = os.path.join(R, "dashboard", "assets")
os.makedirs(ASSETS, exist_ok=True); os.makedirs(os.path.join(R, "docs"), exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
BLUE, ORANGE, GREY, GREEN, RED = "#0072B2", "#D55E00", "#8c8c8c", "#009E73", "#CC79A7"


def jload(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


tasks = jload(os.path.join(Q, "tasks.json"), [])
state = jload(os.path.join(Q, "state.json"), {})
errors = []


def delta_table(rows, metrics, base="base"):
    """per condition: (mean change vs base, CI low, CI high, n) per metric."""
    b = {r["scene"]: r for r in rows if r["condition"] == base}
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    rng = np.random.default_rng(0); out = {}
    for c in conds:
        if c == base:
            continue
        sub = [r for r in rows if r["condition"] == c and r["scene"] in b]
        out[c] = {}
        for m in metrics:
            d = np.array([r[m] - b[r["scene"]][m] for r in sub], float); d = d[np.isfinite(d)]
            if len(d) < 2:
                out[c][m] = (np.nan, np.nan, np.nan, len(d)); continue
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(1000)])
            out[c][m] = (d.mean(), *np.percentile(bs, [2.5, 97.5]), len(d))
    return out


def bar_deltas(dt, conds, metrics, labels, colors, path, title):
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 0.3 * len(conds) + 1.3), sharey=True)
    axs = np.atleast_1d(axs); y = np.arange(len(conds))
    for ax, m, lab in zip(axs, metrics, labels):
        mu = np.array([dt[c][m][0] for c in conds]); lo = np.array([dt[c][m][1] for c in conds]); hi = np.array([dt[c][m][2] for c in conds])
        ax.barh(y, mu, color=[colors.get(c, GREY) for c in conds], height=0.7)
        ok = np.isfinite(lo)
        ax.errorbar(mu[ok], y[ok], xerr=[(mu - lo)[ok], (hi - mu)[ok]], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
        ax.axvline(0, color="black", lw=0.6); ax.set_xlabel(lab)
    axs[0].set_yticks(y); axs[0].set_yticklabels(conds); axs[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=9); fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def md_delta(dt, conds, metrics):
    s = "| condition | n | " + " | ".join(f"change in {m} [95% CI]" for m in metrics) + " |\n|---|---|" + "---|" * len(metrics) + "\n"
    for c in conds:
        s += f"| {c} | {dt[c][metrics[0]][3]} | " + " | ".join(
            f"{dt[c][m][0]:+.3f} [{dt[c][m][1]:+.3f}, {dt[c][m][2]:+.3f}]" for m in metrics) + " |\n"
    return s


def rows_of(res, name):
    J = jload(os.path.join(res, name)) or jload(os.path.join(res, name + ".partial"))
    return (J or {}).get("rows", []), (J or {})


def plot_causal(tid, res, out):
    rows, _ = rows_of(res, "causal_v2.json")
    if not rows:
        return "", []
    met = ["slope", "epe_static", "epe_moving", "sep_ratio"]; dt = delta_table(rows, met)
    order = [c for c in ["abl5", "rand5_r0", "rand5_r1", "rand5_r2", "amp5_a1.5", "amp5_a2", "amp5_a3", "amprand_a2_r0", "amprand_a2_r1",
                         "amprand_a2_r2", "abl5_sfonly", "abl5_nonsf", "abl5_tap0", "abl5_tap1", "abl5_tap2", "abl5_tap3",
                         "amp5_a2_sfonly", "amp5_a2_nonsf"] if c in dt]
    col = {c: (ORANGE if c.startswith("abl5") else BLUE if c.startswith("amp5") else GREY) for c in order}
    f1 = os.path.join(out, "causal_deltas.png")
    bar_deltas(dt, order, met, ["change in slope (object motion recovered)", "change in static EPE (px)", "change in moving EPE (px)",
                                "change in separation ratio"], col, f1,
               f"Five supply heads: ablation (orange), amplification (blue), random heads (grey); {len(set(r['scene'] for r in rows))} scenes")
    f2 = os.path.join(out, "causal_amplification.png")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    key = {0: "abl5", 1: None, 1.5: "amp5_a1.5", 2: "amp5_a2", 3: "amp5_a3"}
    for ax, m, lab in zip(axs, ["slope", "epe_static"], ["change in slope", "change in static EPE (px)"]):
        aa = [a for a in key if key[a] is None or key[a] in dt]
        ax.plot(aa, [0 if key[a] is None else dt[key[a]][m][0] for a in aa], "-o", color=BLUE, ms=3, label="five supply heads")
        rr = [dt[f"amprand_a2_r{r}"][m][0] for r in range(3) if f"amprand_a2_r{r}" in dt]
        if rr:
            ax.scatter([2] * len(rr), rr, color=GREY, s=10, label="random 5 heads, x2")
        ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("head output scale (0 = mean ablation)"); ax.set_ylabel(lab)
    axs[0].legend(frameon=False, fontsize=6); fig.tight_layout(); fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
    return md_delta(dt, order, met), [f1, f2]


def plot_subspace(tid, res, out):
    rows, J = rows_of(res, "subspace_v2.json")
    if not rows:
        return "", []
    met = ["slope", "epe_moving", "epe_static_obj", "epe_bg"]; dt = delta_table(rows, met)
    order = [c for c in dict.fromkeys(r["condition"] for r in rows) if c != "base"]
    col = {c: (BLUE if c.startswith("leace") else ORANGE if c.startswith("steer_") else GREY) for c in order}
    f1 = os.path.join(out, "subspace_deltas.png")
    bar_deltas(dt, order, met, ["change in slope", "change in moving EPE (px)", "change in static-object EPE (px)",
                                "change in background EPE (px)"], col, f1,
               f"Motion subspace: LEACE erasure (blue), steering (orange), random controls (grey); {len(set(r['scene'] for r in rows))} scenes")
    md = md_delta(dt, order, met)
    if J.get("fit"):
        md += "\nEraser fits (training fold; linear R^2 of 'moving' before and after erasure):\n\n| fit | before | after |\n|---|---|---|\n"
        md += "".join(f"| {k} | {v['r2_before']:.3f} | {v['r2_after']:.3f} |\n" for k, v in J["fit"].items())
    return md, [f1]


def layer_means(P, key, split=None):
    rows = P.get("rows", []) if isinstance(P, dict) else []
    by = {}
    for r in rows:
        if split and r.get("split") != split:
            continue
        v = r.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            by.setdefault(r["layer"], []).append(float(v))
    xs = sorted(by); return xs, [float(np.mean(by[x])) for x in xs]


def plot_viewsweep(tid, res, out):
    files = sorted(glob.glob(os.path.join(res, "vs_*_results.json")))
    md = ""; figs = []
    if files:
        cols = {"2": RED, "3": ORANGE, "4": GREEN, "7": BLUE}
        J = {os.path.basename(f).split("_")[-2].rstrip("v"): jload(f, {}) for f in files}
        panels = [("motion_auc_v0_objects", None, "moving vs static AUC, view 0 (all)"),
                  ("motion_auc_v0_objects", "epi_aligned", "moving vs static AUC, epipolar-aligned"),
                  ("pair_auc_obj", None, "same-body pair AUC (object pairs)"),
                  ("objmotion_slope", None, "probe object-motion slope")]
        fig, axs = plt.subplots(1, 4, figsize=(12, 2.6))
        tab = {}
        for nv, j in sorted(J.items(), key=lambda t: int(t[0]) if t[0].isdigit() else 99):
            for ax, (k, sp, lab) in zip(axs, panels):
                xs, ys = layer_means(j.get("P", {}), k, sp)
                if xs:
                    ax.plot(xs, ys, "-o", ms=2.5, color=cols.get(nv, GREY), label=f"{nv} views")
                    tab.setdefault(nv, {})[lab] = (ys[0], ys[-1])
                ax.set_title(lab, fontsize=7.5, loc="left"); ax.set_xlabel("layer")
        axs[0].legend(frameon=False, fontsize=6); fig.tight_layout()
        f1 = os.path.join(out, "viewsweep_probes.png"); fig.savefig(f1, bbox_inches="tight"); plt.close(fig); figs.append(f1)
        md += "| views | " + " | ".join(f"{lab} (first / last layer)" for _, _, lab in panels) + " |\n|---|" + "---|" * len(panels) + "\n"
        for nv in sorted(tab, key=int):
            md += f"| {nv} | " + " | ".join(f"{tab[nv][lab][0]:.3f} / {tab[nv][lab][1]:.3f}" if lab in tab[nv] else "" for _, _, lab in panels) + " |\n"
    s3 = sorted(glob.glob(os.path.join(res, "step3_*v.json")))
    if s3:
        md += "\n| views | output separation ratio, frame 0 to 23 (median over scenes) | n scenes |\n|---|---|---|\n"
        for f in s3:
            s = jload(f, {}); nv = os.path.basename(f).split("_")[-1].replace("v.json", "")
            rows = s.get("rows", s) if isinstance(s, dict) else s
            last = [r for r in (rows or []) if isinstance(r, dict) and r.get("target_frame") == 23]
            vals = [r.get("sep_pred", np.nan) / r["sep_gt"] for r in last if isinstance(r.get("sep_gt"), (int, float)) and r["sep_gt"] > 1]
            md += f"| {nv} | {np.nanmedian(vals) if vals else float('nan'):.3f} | {len(vals)} |\n"
    return md, figs


def generic(tid, res, out):
    """Figures and markdown summaries the task produced itself."""
    md = ""; figs = []
    for f in sorted(glob.glob(os.path.join(res, "**", "*"), recursive=True)):
        rel = os.path.relpath(f, res)
        if f.lower().endswith((".png", ".gif", ".jpg", ".svg")) and os.path.getsize(f) < 15e6:
            dst = os.path.join(out, rel.replace(os.sep, "__")); shutil.copy(f, dst); figs.append(dst)
        elif f.endswith("_summary.md") or os.path.basename(f).lower() == "summary.md":
            md += f"\n**{rel}**\n\n" + open(f).read()[:20000] + "\n"
    return md, figs


PLOTTERS = {"t4_causal": plot_causal, "t5_subspace": plot_subspace, "t3_viewsweep": plot_viewsweep, "v3_viewsweep": plot_viewsweep}
LABEL = {"done": "done", "running": "RUNNING", "failed": "FAILED", "waiting_gate": "waiting for approval", "pending": "queued"}

now = time.strftime("%Y-%m-%d %H:%M:%S")
lines = ["# E5 experiment dashboard", "", f"Last updated {now} (server time). Regenerated after every queue event.", "",
         "| # | run | status | attempts | started | finished |", "|---|---|---|---|---|---|"]
for i, t in enumerate(tasks):
    s = state.get(t["id"], {}); st = s.get("status", "pending")
    gate = f" (gate `{t['gate']}`)" if t.get("gate") and st == "waiting_gate" else ""
    lines.append(f"| {i+1} | [{t['id']}](#{t['id'].lower()}): {t.get('title', '')} | {LABEL.get(st, st)}{gate} | {s.get('attempts', 0)} | "
                 f"{s.get('started', '')} | {s.get('finished', '') if st in ('done', 'failed') else ''} |")
hb = os.path.join(Q, "heartbeat")
if os.path.exists(hb):
    lines += ["", f"Supervisor heartbeat: {open(hb).read().strip()}"]
sections = []
for t in tasks:
    tid = t["id"]; s = state.get(tid, {}); st = s.get("status", "pending")
    res = os.path.join(Q, "results", tid)
    if st not in ("done", "failed", "running") or not os.path.isdir(res):
        continue
    out = os.path.join(ASSETS, tid); shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
    md, figs = "", []
    try:
        fn = next((f for k, f in PLOTTERS.items() if tid.startswith(k)), None)
        if fn:
            md, figs = fn(tid, res, out)
    except Exception:
        errors.append(f"{tid}: {traceback.format_exc()[-500:]}")
    try:
        gmd, gfigs = generic(tid, res, out); md += gmd; figs += gfigs
    except Exception:
        errors.append(f"{tid} (task figures): {traceback.format_exc()[-500:]}")
    sec = [f"## {tid}", "", t.get("title", ""), "",
           f"Status: **{LABEL.get(st, st)}**, attempt {s.get('attempts', 0)}, started {s.get('started', '')}"
           + (f", finished {s.get('finished', '')}" if st in ("done", "failed") else "") + ". "
           f"[Raw results](../results/queue/{tid}/), [log](../results/queue/{tid}/task.log).", ""]
    if st == "running":
        sec.append("Partial results so far; the run is still going.\n")
    sec += [f"![{os.path.basename(f)}](assets/{tid}/{os.path.basename(f)})" for f in figs if not f.endswith(".gif")]
    gifs = [f for f in figs if f.endswith(".gif")]
    if gifs:
        sec += ["", " ".join(f'<img src="assets/{tid}/{os.path.basename(f)}" width="200">' for f in gifs)]
    if md:
        sec += ["", md]
    sections.append("\n".join(sec))
page = "\n".join(lines) + "\n\n" + "\n\n".join(sections)
if errors:
    page += "\n\n## Dashboard build notes\n\n" + "\n".join(f"- `{e[:300]}`" for e in errors)
open(os.path.join(R, "dashboard", "README.md"), "w").write(page + "\n")

# static HTML copy for GitHub Pages (docs/)
dassets = os.path.join(R, "docs", "assets"); shutil.rmtree(dassets, ignore_errors=True); shutil.copytree(ASSETS, dassets)
try:
    import markdown
    body = markdown.markdown(page, extensions=["tables"])
except Exception:
    body = "<pre>" + html.escape(page) + "</pre>"
body = body.replace("../results/queue/", REPO_URL + "/tree/main/results/queue/")
open(os.path.join(R, "docs", "index.html"), "w").write(
    "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='300'><title>E5 dashboard</title>"
    "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}img{max-width:100%}"
    "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px 6px;font-size:13px}</style></head><body>" + body + "</body></html>")
print("dashboard built:", len(tasks), "tasks,", len(sections), "sections,", len(errors), "errors")
