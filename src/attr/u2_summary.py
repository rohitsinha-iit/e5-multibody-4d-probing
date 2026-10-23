#!/usr/bin/env python3
"""U2 summary (re-runnable): tables with 95% scene-bootstrap CIs (2000 resamples, seed 0), the U2 confirmatory
statistics (raw p-values; Holm over the full family is applied by the lead), motion-budget shares, real-data and
pose-set tables, and figure drafts F1 and F2.

Inputs: results/u2/rows_kubric.csv, rows_real.csv, rows_pose.csv (u2_score.py).
Outputs: results/u2/summary_u2.json, summary_u2.md, confirmatory_u2.json, fig_F1_factorial.png, fig_F2_crossfamily.png.

Confirmatory tests as frozen in results/attr_preregistration.json (boundary rule primary = excl8; variant D unless
stated; statistic = scene mean; unit = base scene):
  T1a  OpenD4RT r3 s_w < 0.9, one-sided bootstrap; the frozen text pools "v2 test + v3float transfer", so the
       confirmatory p uses the pooled scenes; test-only and transfer-only p are reported alongside (descriptive).
  T1b  TrackCraft3R condition (ii) (DA3 inputs) s_w < 0.9, same test.
  T1 validity  lifted CoTracker3 and lifted RAFT must give s_w >= 0.9 (scene mean on the same scenes) for T1 to be read;
       lifted RAFT primary = chained over the 7 views (E5 "chained RAFT"), direct 0 -> 23 reported alongside.
  T7   I = alpha_D - alpha_C for Any4D on TEST scenes identifiable in D, two-sided paired scene bootstrap; decision rule
       |I| >= 0.10 effect (negative supports account K), |I| < 0.05 with CI including 0 = no interaction (account R),
       otherwise inconclusive; repeated with s_w (descriptive); transfer reported alongside (descriptive).
  T8   dRPE-R = RPE-R(D) - RPE-R(B) > 0 for Any4D on TEST scenes, one-sided paired bootstrap; VGGT descriptive.
Bootstrap p-values: one-sided p = (#{b : stat_b on the wrong side of the null} + 1) / (B + 1); two-sided p =
min(1, 2 min(p_low, p_high)).
"""
import csv, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE); sys.path.append(os.path.dirname(HERE))
import u2_common as u  # noqa: E402

NB, SEED = 2000, 0
BASE_METRICS = ["alpha", "beta", "slope2d", "beta_static", "epe_static", "epe_moving", "sep_ratio", "s_w", "static_residual",
                "pose_ate", "pose_rpe_t", "pose_rpe_r", "spur_rot_deg", "spur_trans_rel", "v7_pose_ate", "v7_pose_rpe_t",
                "v7_pose_rpe_r", "v7_spur_rot_deg", "v7_spur_trans_rel", "budget_geometry", "budget_flow", "budget_camera"]
PREFIXES = ["", "r1_", "r2_", "r2gtK_", "nooff_", "direct_"]
MODEL_ORDER = ["any4d", "vggt", "opend4rt", "4rc", "tc3r_gt", "tc3r_da3", "tc3r_identity", "cotracker3", "raft", "more"]


def num(x):
    if x in (None, "", "None"):
        return np.nan
    if x in ("True", "False"):
        return 1.0 if x == "True" else 0.0
    try:
        return float(x)
    except ValueError:
        return x


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def boot(v, stat=np.mean):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, est=np.nan, lo=np.nan, hi=np.nan)
    rs = np.random.default_rng(SEED); b = np.array([stat(v[rs.integers(0, len(v), len(v))]) for _ in range(NB)])
    return dict(n=int(len(v)), est=float(stat(v)), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))


def boot_dist(v, stat=np.mean):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rs = np.random.default_rng(SEED)
    return v, np.array([stat(v[rs.integers(0, len(v), len(v))]) for _ in range(NB)]) if len(v) else np.array([])


def p_one_sided_below(v, thr):
    v, b = boot_dist(v)
    return float((np.sum(b >= thr) + 1) / (NB + 1)) if len(v) else np.nan


def p_one_sided_above(v, thr):
    v, b = boot_dist(v)
    return float((np.sum(b <= thr) + 1) / (NB + 1)) if len(v) else np.nan


def p_two_sided(v, thr=0.0):
    v, b = boot_dist(v)
    if not len(v):
        return np.nan
    lo = (np.sum(b <= thr) + 1) / (NB + 1); hi = (np.sum(b >= thr) + 1) / (NB + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def col(rows, k, ident_only=False, ident_key="identifiable"):
    out = []
    for r in rows:
        x = r.get(k, np.nan)
        if ident_only and not (r.get(ident_key, 0) == 1.0):
            x = np.nan
        out.append(x if isinstance(x, float) else np.nan)
    return np.array(out, float)


def kubric_tables(K):
    T = []
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in K)]
    for m in models:
        for sp in ("fit", "test", "transfer"):
            for v in "ABCD":
                for bd in ("excl8", "none"):
                    rr = sel(K, model=m, split=sp, variant=v, boundary=bd)
                    if not rr:
                        continue
                    ent = dict(model=m, split=sp, variant=v, boundary=bd, n_scenes=len(rr),
                               n_identifiable=int(np.nansum(col(rr, "identifiable"))), metrics={})
                    for pre in PREFIXES:
                        for k in BASE_METRICS:
                            key = pre + k
                            if key not in rr[0] and not any(key in r for r in rr):
                                continue
                            idn = k in ("alpha", "beta") and v == "D"
                            x = col(rr, key, ident_only=idn, ident_key=pre + "identifiable")
                            if not np.isfinite(x).any():
                                continue
                            ent["metrics"][key] = dict(mean=boot(x), median=boot(x, np.median))
                    T.append(ent)
    return T


def paired(K, m, sp, bd, va, vb, key, ident_from=None):
    A = {r["seed"]: r for r in sel(K, model=m, split=sp, variant=va, boundary=bd)}
    Bv = {r["seed"]: r for r in sel(K, model=m, split=sp, variant=vb, boundary=bd)}
    seeds = sorted(set(A) & set(Bv)); d = []
    for s in seeds:
        if ident_from is not None and not ({"a": A, "b": Bv}[ident_from][s].get("identifiable", 0) == 1.0):
            continue
        xa, xb = A[s].get(key, np.nan), Bv[s].get(key, np.nan)
        if isinstance(xa, float) and isinstance(xb, float) and np.isfinite(xa) and np.isfinite(xb):
            d.append(xa - xb)
    return np.array(d, float)


def derived(K):
    D = []
    for m in [m for m in MODEL_ORDER if any(r["model"] == m for r in K)]:
        for sp in ("fit", "test", "transfer"):
            for bd in ("excl8", "none"):
                ent = dict(model=m, split=sp, boundary=bd, metrics={})
                for name, va, vb, key, idf in (("I_alpha_D_minus_C", "D", "C", "alpha", "a"), ("I_sw_D_minus_C", "D", "C", "s_w", None),
                                               ("dRPE_R_D_minus_B", "D", "B", "pose_rpe_r", None), ("dRPE_T_D_minus_B", "D", "B", "pose_rpe_t", None),
                                               ("dATE_D_minus_B", "D", "B", "pose_ate", None),
                                               ("spur_rot_C_minus_A", "C", "A", "spur_rot_deg", None), ("spur_trans_C_minus_A", "C", "A", "spur_trans_rel", None),
                                               ("v7_dRPE_R_D_minus_B", "D", "B", "v7_pose_rpe_r", None),
                                               ("v7_spur_rot_C_minus_A", "C", "A", "v7_spur_rot_deg", None)):
                    d = paired(K, m, sp, bd, va, vb, key, idf)
                    if len(d):
                        ent["metrics"][name] = dict(mean=boot(d), median=boot(d, np.median))
                if ent["metrics"]:
                    D.append(ent)
    return D


def scenes_vals(K, m, splits, v, key, bd="excl8"):
    return np.concatenate([col(sel(K, model=m, split=s, variant=v, boundary=bd), key) for s in splits]) if splits else np.array([])


def confirmatory(K):
    C = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), preregistration_sha256=None, boundary="excl8 (primary); none in *_noboundary",
             bootstrap=dict(resamples=NB, seed=SEED, stat="scene mean"), note="raw p-values; Holm over the full family by the lead")
    import hashlib
    C["preregistration_sha256"] = hashlib.sha256(open(u.PREREG, "rb").read()).hexdigest()
    for bd, suf in (("excl8", ""), ("none", "_noboundary")):
        for T, m in (("T1a", "opend4rt"), ("T1b", "tc3r_da3")):
            pooled = scenes_vals(K, m, ["test", "transfer"], "D", "s_w", bd)
            ent = dict(model=m, readout="r3 world displacement" if m == "opend4rt" else "condition (ii) DA3 inputs, stretch, no t0 registration",
                       H1="mean s_w < 0.9", sides="one-sided", scenes="v2 test + v3float transfer (pooled, as frozen)",
                       s_w=boot(pooled), p=p_one_sided_below(pooled, 0.9))
            for sp in ("test", "transfer"):
                x = scenes_vals(K, m, [sp], "D", "s_w", bd); ent[f"{sp}_only"] = dict(s_w=boot(x), p=p_one_sided_below(x, 0.9))
            C[T + suf] = ent
        val = {}
        for m, key in (("cotracker3", "s_w"), ("raft", "s_w"), ("raft", "direct_s_w")):
            x = scenes_vals(K, m, ["test", "transfer"], "D", key, bd)
            b = boot(x); val[f"{m}:{key}"] = dict(s_w=b, passes=bool(np.isfinite(b["est"]) and b["est"] >= 0.9),
                                                    test_only=boot(scenes_vals(K, m, ["test"], "D", key, bd)))
        prim = [val["cotracker3:s_w"]["passes"], val["raft:s_w"]["passes"]]
        C["T1_validity" + suf] = dict(rule="lifted CoTracker3 and lifted RAFT s_w >= 0.9 for T1 to be read (RAFT primary = chained)",
                                      lifted=val, T1_readable=bool(all(prim)))
        for sp in ("test", "transfer"):
            I = paired(K, "any4d", sp, bd, "D", "C", "alpha", "a"); b = boot(I)
            if np.isfinite(b["est"]):
                incl0 = b["lo"] <= 0 <= b["hi"]
                dec = "effect" if abs(b["est"]) >= 0.10 else ("no interaction" if (abs(b["est"]) < 0.05 and incl0) else "inconclusive")
            else:
                dec = "undefined"
            ent = dict(model="any4d", I="alpha_D - alpha_C, test scenes identifiable in D", I_stat=b, p_two_sided=p_two_sided(I),
                       decision=dec, account_hint=("K" if dec == "effect" and b["est"] < 0 else ("R" if dec == "no interaction" else "")),
                       s_w_repeat=dict(I_sw=boot(paired(K, "any4d", sp, bd, "D", "C", "s_w")), p_two_sided=p_two_sided(paired(K, "any4d", sp, bd, "D", "C", "s_w"))))
            C[("T7" if sp == "test" else "T7_transfer_descriptive") + suf] = ent
            for m in ("any4d", "vggt"):
                d = paired(K, m, sp, bd, "D", "B", "pose_rpe_r")
                ent = dict(model=m, stat="dRPE-R = RPE-R(D) - RPE-R(B), deg", H1="mean > 0", dRPE_R=boot(d), p_one_sided=p_one_sided_above(d, 0.0),
                           role="confirmatory" if (m == "any4d" and sp == "test") else "descriptive")
                C[("T8" if m == "any4d" else "T8_vggt") + ("" if sp == "test" else "_transfer") + suf] = ent
    return C


def real_tables(Rr):
    T = []
    keys = ["slope", "alpha", "beta", "beta_static", "epe_static", "epe_moving", "sep_ratio"]
    for m in sorted(set(r["model"] for r in Rr)):
        for unit in sorted(set(r["unit"] for r in Rr if r["model"] == m)):
            rr = sel(Rr, model=m, unit=unit); ent = dict(model=m, unit=unit, n=len(rr), metrics={})
            for pre in ("", "r2_", "r4own_"):
                for k in keys:
                    x = col(rr, pre + k, ident_only=(k in ("alpha", "beta")), ident_key=pre + "identifiable")
                    if np.isfinite(x).any():
                        ent["metrics"][pre + k] = dict(mean=boot(x), median=boot(x, np.median))
            T.append(ent)
    return T


def pose_tables(P):
    T = []
    for m in [m for m in MODEL_ORDER if any(r["model"] == m for r in P)]:
        for s in ("sintel", "tum", "bonn"):
            rr = sel(P, model=m, set=s)
            if rr:
                T.append(dict(model=m, set=s, n_windows=len(rr), metrics={k: dict(mean=boot(col(rr, k)), median=boot(col(rr, k), np.median))
                                                                         for k in ("ate", "rpe_t", "rpe_r")}))
    return T


def fmt(b, nd=3):
    return "n/a" if not np.isfinite(b.get("est", np.nan)) else f"{b['est']:.{nd}f} [{b['lo']:.{nd}f}, {b['hi']:.{nd}f}]"


def md(S, C):
    L = ["# U2 behaviour summary (auto-generated by src/attr/u2_summary.py)", "", f"Written {S['written']}. Bootstrap: 2000 scene resamples, seed 0, 95% percentile CI; mean [CI] unless marked.", ""]
    L += ["## Confirmatory statistics (raw p; Holm by the lead)", ""]
    for k in ("T1a", "T1b"):
        e = C.get(k)
        if e:
            L.append(f"- {k} ({e['model']}): mean s_w {fmt(e['s_w'])}, n {e['s_w']['n']}, one-sided p {e['p']:.4g} (test only p {e['test_only']['p']:.4g}, transfer only p {e['transfer_only']['p']:.4g})")
    v = C.get("T1_validity")
    if v:
        L.append(f"- T1 validity: readable = {v['T1_readable']}; " + "; ".join(f"{k} {fmt(x['s_w'])}" for k, x in v["lifted"].items()))
    for k in ("T7", "T7_transfer_descriptive"):
        e = C.get(k)
        if e:
            L.append(f"- {k}: I {fmt(e['I_stat'])}, n {e['I_stat']['n']}, two-sided p {e['p_two_sided']:.4g}, decision {e['decision']}; s_w repeat {fmt(e['s_w_repeat']['I_sw'])}")
    for k in ("T8", "T8_vggt", "T8_transfer", "T8_vggt_transfer"):
        e = C.get(k)
        if e:
            L.append(f"- {k} ({e['role']}): dRPE-R {fmt(e['dRPE_R'])}, n {e['dRPE_R']['n']}, one-sided p {e['p_one_sided']:.4g}")
    L += ["", "## Kubric factorial (boundary rule excl8), mean [95% CI]", ""]
    show = ["alpha", "beta", "slope2d", "s_w", "static_residual", "epe_static", "epe_moving", "sep_ratio", "pose_rpe_r", "pose_ate", "spur_rot_deg", "spur_trans_rel"]
    L.append("| model | split | V | n | " + " | ".join(show) + " |"); L.append("|" + "---|" * (4 + len(show)))
    for e in S["kubric"]:
        if e["boundary"] != "excl8":
            continue
        L.append(f"| {e['model']} | {e['split']} | {e['variant']} | {e['n_scenes']} | " + " | ".join(fmt(e["metrics"][k]["mean"], 2) if k in e["metrics"] else "" for k in show) + " |")
    L += ["", "## Paired contrasts (excl8), mean [95% CI]", ""]
    dk = ["I_alpha_D_minus_C", "I_sw_D_minus_C", "dRPE_R_D_minus_B", "dATE_D_minus_B", "spur_rot_C_minus_A", "spur_trans_C_minus_A"]
    L.append("| model | split | " + " | ".join(dk) + " |"); L.append("|" + "---|" * (2 + len(dk)))
    for e in S["derived"]:
        if e["boundary"] == "excl8":
            L.append(f"| {e['model']} | {e['split']} | " + " | ".join(fmt(e["metrics"][k]["mean"], 3) if k in e["metrics"] else "" for k in dk) + " |")
    L += ["", "## Motion budget (scene median of phi_c / sum phi; median [95% CI] over scenes; excl8)", ""]
    L.append("| model | split | V | n | geometry | flow | camera |"); L.append("|---|---|---|---|---|---|---|")
    for e in S["kubric"]:
        if e["boundary"] == "excl8" and "budget_flow" in e["metrics"]:
            g = e["metrics"]
            L.append(f"| {e['model']} | {e['split']} | {e['variant']} | {g['budget_flow']['median']['n']} | {fmt(g['budget_geometry']['median'], 2)} | {fmt(g['budget_flow']['median'], 2)} | {fmt(g['budget_camera']['median'], 2)} |")
    L += ["", "## Real data (E5 protocol), mean [95% CI]", ""]
    rk = ["slope", "alpha", "beta", "epe_static", "epe_moving", "sep_ratio", "r4own_slope", "r2_slope"]
    L.append("| model | unit | n | " + " | ".join(rk) + " |"); L.append("|" + "---|" * (3 + len(rk)))
    for e in S["real"]:
        L.append(f"| {e['model']} | {e['unit']} | {e['n']} | " + " | ".join(fmt(e["metrics"][k]["mean"], 2) if k in e["metrics"] else "" for k in rk) + " |")
    L += ["", "## Pose sets (8-frame windows, Sim(3) per window), mean [95% CI]", "", "| model | set | windows | ATE | RPE-T | RPE-R (deg) |", "|---|---|---|---|---|---|"]
    for e in S["pose"]:
        g = e["metrics"]; L.append(f"| {e['model']} | {e['set']} | {e['n_windows']} | {fmt(g['ate']['mean'], 4)} | {fmt(g['rpe_t']['mean'], 4)} | {fmt(g['rpe_r']['mean'], 3)} |")
    return "\n".join(L) + "\n"


def figures(K, S, C):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}
    lab = dict(any4d="Any4D", vggt="VGGT", opend4rt="OpenD4RT", **{"4rc": "4RC"}, tc3r_gt="TC3R (i)", tc3r_da3="TC3R (ii)",
               tc3r_identity="TC3R (iii)", cotracker3="CoTracker3 (lifted)", raft="RAFT (lifted)", more="MoRe")
    # F1: factorial 2x2 for the reconstruction models on the test split + budget
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ms = [m for m in ("any4d", "vggt", "opend4rt", "4rc") if sel(K, model=m, split="test", variant="D")]
    for i, m in enumerate(ms):
        for j, (va, key, idf) in enumerate((("C", "alpha", None), ("D", "alpha", "D"))):
            rr = sel(K, model=m, split="test", variant=va, boundary="excl8"); x = col(rr, key, ident_only=(idf == "D"))
            b = boot(x); xx = i + (j - 0.5) * 0.3
            ax[0].errorbar([xx], [b["est"]], yerr=[[b["est"] - b["lo"]], [b["hi"] - b["est"]]], fmt="o" if va == "C" else "s",
                           color="C0" if va == "C" else "C3", label=(f"{va} (camera {'frozen' if va == 'C' else 'moving'})" if i == 0 else None))
    ax[0].axhline(1, color="0.6", lw=0.8, ls="--"); ax[0].set_xticks(range(len(ms))); ax[0].set_xticklabels([lab[m] for m in ms])
    ax[0].set_ylabel("alpha (own object motion recovered)"); ax[0].set_title("a  object motion: C vs D (test)"); ax[0].legend(frameon=False, fontsize=8)
    ms2 = [m for m in ("any4d", "vggt", "opend4rt", "4rc", "more") if sel(K, model=m, split="test", variant="B")]
    for i, m in enumerate(ms2):
        for j, va in enumerate("BD"):
            b = boot(col(sel(K, model=m, split="test", variant=va, boundary="excl8"), "pose_rpe_r")); xx = i + (j - 0.5) * 0.3
            ax[1].errorbar([xx], [b["est"]], yerr=[[b["est"] - b["lo"]], [b["hi"] - b["est"]]], fmt="o", color="C0" if va == "B" else "C3",
                           label=(f"{va} (objects {'frozen' if va == 'B' else 'moving'})" if i == 0 else None))
    ax[1].set_xticks(range(len(ms2))); ax[1].set_xticklabels([lab[m] for m in ms2], rotation=20); ax[1].set_ylabel("RPE-R (deg)")
    ax[1].set_title("b  camera error: B vs D (test)"); ax[1].legend(frameon=False, fontsize=8)
    bars = []
    for m in ("any4d", "4rc"):
        for va in "CD":
            e = [x for x in S["kubric"] if x["model"] == m and x["split"] == "test" and x["variant"] == va and x["boundary"] == "excl8"]
            if e and "budget_flow" in e[0]["metrics"]:
                g = e[0]["metrics"]; bars.append((f"{lab[m]} {va}", [g[f"budget_{c}"]["median"]["est"] for c in ("geometry", "flow", "camera")]))
    if bars:
        bot = np.zeros(len(bars))
        for ci, cn in enumerate(("geometry", "flow", "camera")):
            vals = np.array([b[1][ci] for b in bars]); ax[2].bar(range(len(bars)), vals, bottom=bot, label=cn, color=f"C{ci + 4}"); bot += vals
        ax[2].set_xticks(range(len(bars))); ax[2].set_xticklabels([b[0] for b in bars]); ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_ylabel("Shapley share of the deficit (scene median)"); ax[2].set_title("c  motion budget (test)")
    fig.tight_layout(); p = os.path.join(u.U2, "fig_F1_factorial.png"); fig.savefig(p, dpi=150); plt.close(fig); out["F1"] = p
    # F2: cross-family s_w and contamination
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ms = [m for m in ("any4d", "vggt", "opend4rt", "4rc", "tc3r_gt", "tc3r_da3", "cotracker3", "raft") if sel(K, model=m, variant="D")]
    rs = np.random.default_rng(0)
    for i, m in enumerate(ms):
        x = scenes_vals(K, m, ["test", "transfer"], "D", "s_w"); x = x[np.isfinite(x)]
        ax[0].scatter(i + rs.uniform(-0.15, 0.15, len(x)), x, s=6, color="0.6", alpha=0.6)
        b = boot(x); ax[0].errorbar([i], [b["est"]], yerr=[[b["est"] - b["lo"]], [b["hi"] - b["est"]]], fmt="o", color="C3")
    ax[0].axhline(0.9, color="C0", ls="--", lw=0.8); ax[0].axhline(1.0, color="0.6", lw=0.8)
    ax[0].set_xticks(range(len(ms))); ax[0].set_xticklabels([lab[m] for m in ms], rotation=25); ax[0].set_ylabel("world-frame slope s_w (D)")
    ax[0].set_title("a  s_w, test + transfer (mean, 95% CI)"); ax[0].set_ylim(-0.2, 1.6)
    ms2 = [m for m in ("any4d", "vggt", "opend4rt", "4rc", "more") if sel(K, model=m, variant="B")]
    for i, m in enumerate(ms2):
        for j, sp in enumerate(("test", "transfer")):
            b = boot(paired(K, m, sp, "excl8", "D", "B", "pose_rpe_r"))
            ax[1].errorbar([i + (j - 0.5) * 0.3], [b["est"]], yerr=[[b["est"] - b["lo"]], [b["hi"] - b["est"]]], fmt="o", color=f"C{j}",
                           label=sp if i == 0 else None)
    ax[1].axhline(0, color="0.6", lw=0.8); ax[1].set_xticks(range(len(ms2))); ax[1].set_xticklabels([lab[m] for m in ms2])
    ax[1].set_ylabel("dRPE-R = RPE-R(D) - RPE-R(B) (deg)"); ax[1].set_title("b  contamination of camera estimates"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); p = os.path.join(u.U2, "fig_F2_crossfamily.png"); fig.savefig(p, dpi=150); plt.close(fig); out["F2"] = p
    return out


def main():
    K = load(os.path.join(u.U2, "rows_kubric.csv")); Rr = load(os.path.join(u.U2, "rows_real.csv")); P = load(os.path.join(u.U2, "rows_pose.csv"))
    for r in K:
        r["seed"] = int(r["seed"])
    S = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), n_rows=dict(kubric=len(K), real=len(Rr), pose=len(P)),
             kubric=kubric_tables(K), derived=derived(K), real=real_tables(Rr), pose=pose_tables(P))
    C = confirmatory(K)
    u.save_json(os.path.join(u.U2, "summary_u2.json"), S); u.save_json(os.path.join(u.U2, "confirmatory_u2.json"), C)
    open(os.path.join(u.U2, "summary_u2.md"), "w").write(md(S, C))
    try:
        figs = figures(K, S, C)
    except Exception as e:
        import traceback; figs = dict(error=repr(e), tb=traceback.format_exc()[-1500:])
    print(json.dumps(dict(rows=S["n_rows"], figures=figs), default=str))


if __name__ == "__main__":
    main()
