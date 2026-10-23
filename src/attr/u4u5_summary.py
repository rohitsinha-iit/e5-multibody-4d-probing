#!/usr/bin/env python3
"""U4a / U5a / U5b summary on test, transfer and real data (mechanism track). Reads only finished per-item outputs, so
it can be run at any time (partial runs give partial statistics, flagged by n). Statistics follow the frozen
pre-registration (attr_preregistration.json: confirmatory T9, T5p, T10; success rules) and the analysis conventions
frozen in its fit_derived_values block.
Outputs: $E5_BIG/results/u4u5_confirmatory.json (raw p-values; the lead applies Holm),
         $E5_BIG/results/u4u5_summary.json (everything else), figure drafts $E5_BIG/results/u4u5_figs/*.png
"""
import glob, json, os, sys, time
import numpy as np
E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
RES = os.path.join(E5_BIG, "results"); U4A = os.path.join(RES, "u4a"); U5A = os.path.join(RES, "u5a"); U5B = os.path.join(RES, "u5b")
FIG = os.path.join(RES, "u4u5_figs")
B, SEED = 2000, 0
KGRID = [1, 3, 5, 10, 20]
FIVE = ["b2h1", "b22h9", "b17h10", "b14h6", "b8h11"]


def jl(f):
    return json.load(open(f))


def files(pat):
    return [f for f in sorted(glob.glob(pat)) if not f.endswith(".tmp")]


def boot_mean(v, n=B, seed=SEED):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, p_gt0=float("nan"))
    rs = np.random.default_rng(seed); bm = np.array([v[rs.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return dict(mean=float(v.mean()), lo=float(np.quantile(bm, 0.025)), hi=float(np.quantile(bm, 0.975)), n=int(len(v)),
                p_gt0=float((1 + np.sum(bm <= 0)) / (n + 1)))      # one-sided p for H1: mean > 0


def med(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


# ------------------------------------------------------------------ U4a
def joint_rows(screen, split):
    return [jl(f) for f in files(os.path.join(U4A, f"joint_{screen}_{split}", "*.json"))]


def t9_side(screen, split, k, met, leak):
    rows = [r for r in joint_rows(screen, split) if f"k{k}" in r["rows"] and "nulls" in r["rows"][f"k{k}"]]
    if not rows:
        return dict(n_scenes=0)
    obs = med([r["rows"][f"k{k}"][met] for r in rows])
    nn = min(len(r["rows"][f"k{k}"]["nulls"]) for r in rows)
    nulls = np.array([med([r["rows"][f"k{k}"]["nulls"][i][met] for r in rows]) for i in range(nn)])
    lk = med([r["rows"][f"k{k}"][leak] for r in rows])
    out = dict(n_scenes=len(rows), k=k, metric=met, observed_median=obs, null_medians=nulls.tolist(), n_null=int(nn),
               null_pct95=float(np.percentile(nulls, 95)), above_pct95=bool(obs > np.percentile(nulls, 95)),
               p_null_rank=float((1 + np.sum(nulls >= obs)) / (nn + 1)), leakage_name=leak, leakage_median=lk,
               leakage_ok=bool(lk <= 0.2),
               null_leakage_medians=[med([r["rows"][f"k{k}"]["nulls"][i][leak] for r in rows]) for i in range(nn)])
    if met == "tau_O":
        out["observed_median_nb"] = med([r["rows"][f"k{k}"]["tau_O_nb"] for r in rows])
        out["null_medians_nb"] = [med([r["rows"][f"k{k}"]["nulls"][i]["tau_O_nb"] for r in rows]) for i in range(nn)]
    return out


def t9(fz, split):
    ko, kc = fz["k_used"]["objall"], fz["k_used"]["cam"]
    conf = bool(fz["kstar"]["objall"]["kstar"] and fz["kstar"]["cam"]["kstar"])
    O = t9_side("objall", split, ko, "tau_O", "dpose_rel"); C = t9_side("cam", split, kc, "tau_C", "dF_rel")
    if not O.get("n_scenes") or not C.get("n_scenes"):
        return dict(status="pending", P_obj=O, P_cam=C)
    return dict(status="ok", confirmatory=conf,
                note=None if conf else "k* is None for a screen: computed at k_used (fallback k = 20), descriptive (risk 4)",
                P_obj=O, P_cam=C, p=max(O["p_null_rank"], C["p_null_rank"]),
                passes=bool(O["above_pct95"] and C["above_pct95"] and O["leakage_ok"] and C["leakage_ok"]))


def completeness(screen, split):
    rows = joint_rows(screen, split) if split != "fit" else [jl(f) for f in files(os.path.join(U4A, f"joint_{screen}", "*.json"))]
    out = {"n": len(rows)}
    for key in ("completeness", "completeness_nb"):
        vals = {k: med([r["rows"][f"k{k}"].get(key, np.nan) for r in rows if f"k{k}" in r["rows"]]) for k in KGRID}
        if any(np.isfinite(list(vals.values()))):
            out[key] = vals
    return out


def necessity(screen, split):
    rows = [jl(f) for f in files(os.path.join(U4A, f"necessity_{screen}_{split}", "*.json"))]
    return {"n": len(rows), **{f"k{k}": {m: med([r["rows"][f"k{k}"].get(m, np.nan) for r in rows]) for m in
                                         ("tau_O", "tau_C", "kappa", "dF_rel", "dpose_rel", "rpe_r")} for k in KGRID}}


def camwarp(split):
    rows = [jl(f) for f in files(os.path.join(U4A, f"screen_camwarp_{split}", "*.json"))]
    out = {"n": len(rows)}
    if not rows:
        return out
    for n in sorted(rows[0]["rows"], key=lambda x: int(x[1:-1]) if x.endswith("r") else 99):
        e = [r["rows"][n] for r in rows if n in r["rows"]]
        diff = np.array([x["dpose_rel"] - x["floor_dpose_rel"] for x in e])
        out[n] = dict(tau_C=med([x["tau_C"] for x in e]), dpose_rel=med([x["dpose_rel"] for x in e]),
                      floor_dpose_rel=med([x["floor_dpose_rel"] for x in e]), floor_tau_C=med([x["floor_tau_C"] for x in e]),
                      frac_scenes_above_floor=float(np.mean(diff > 0)), above_floor=bool(med(diff) > 0))
    return out


def h3_test(split):
    rows = [jl(f) for f in files(os.path.join(U4A, f"screen_objall_{split}", "*.json"))]
    if not rows:
        return {"n": 0}
    names = sorted({k for r in rows for k in r["rows"] if k not in ("ALL_HEADS", "FINAL_RESID") and not k.endswith("r")})
    m = {n: med([r["rows"][n]["tau_O"] for r in rows if n in r["rows"]]) for n in names}
    order = sorted(names, key=lambda n: -m[n])
    ranks = {h: order.index(h) + 1 for h in FIVE if h in order}
    return {"n": len(rows), "five_rank_heads": ranks, "all_five_top20": all(v <= 20 for v in ranks.values()) and len(ranks) == 5}


# ------------------------------------------------------------------ U5a
def jl_split(split, d):
    pre = "v2_" if split == "test" else "v3float_"
    return {k: v for k, v in d.items() if k.startswith(pre)}


def u5a_stats(split_or_ds):
    kub = split_or_ds in ("test", "transfer")
    ev = "kubric" if kub else split_or_ds
    E = {jl(f)["item"]: jl(f) for f in files(os.path.join(U5A, ev, "edits", "*.json"))}
    V = {jl(f)["item"]: jl(f) for f in files(os.path.join(U5A, ev, "v", "*.json"))}
    if kub:
        E, V = jl_split(split_or_ds, E), jl_split(split_or_ds, V)
    keys = (["alpha", "beta", "s_w", "epe_moving", "epe_static", "alpha_nb", "beta_nb", "epe_moving_nb", "epe_static_nb", "rpe_r"]
            if kub else ["slope", "epe_moving", "epe_static"])
    variants = sorted({v for e in E.values() for v in e["rows"] if v != "base"}) + (["v"] if V else [])
    out = {"n_items": len(E), "variants": {}}
    per = {}
    for name, e in E.items():
        b = e["rows"]["base"]["r0"]
        if kub and not b.get("identifiable"):
            continue
        for vn in variants:
            rr = V[name]["rows"]["v"] if vn == "v" and name in V else e["rows"].get(vn) if vn != "v" else None
            if not rr:
                continue
            d = {k: float(np.nanmean([r[k] - b[k] for r in rr.values()])) for k in keys if k in b}
            d["epe_static_ratio"] = float(np.nanmean([r["epe_static"] for r in rr.values()]) / b["epe_static"] - 1) if b["epe_static"] else float("nan")
            per.setdefault(vn, {})[name] = d
    out["n_used"] = len({n for v in per.values() for n in v})
    for vn, dd in per.items():
        out["variants"][vn] = {k: boot_mean([d[k] for d in dd.values()]) for k in list(keys) + ["epe_static_ratio"] if k in next(iter(dd.values()))}
    if kub:
        for g in ("1.5", "2"):
            i, r = per.get(f"i_g{g}", {}), per.get(f"iii_g{g}", {})
            common = sorted(set(i) & set(r))
            out[f"T5p_g{g}"] = dict(boot_mean([i[s]["alpha"] - r[s]["alpha"] for s in common]),
                                    definition="d_s = d alpha (i) - mean_r d alpha (iii); one-sided p for mean d > 0")
            vi, vii = out["variants"].get(f"i_g{g}"), out["variants"].get(f"ii_g{g}")
            if vi and vii:
                out[f"success_U5a_g{g}"] = dict(
                    i_alpha_up_010=vi["alpha"]["mean"] >= 0.10, i_beta_change_le_003=abs(vi["beta"]["mean"]) <= 0.03,
                    i_moving_epe_le_1px=vi["epe_moving"]["mean"] <= 1.0, i_static_epe_le_5pct=vi["epe_static_ratio"]["mean"] <= 0.05,
                    ii_beta_change_ge_010=vii["beta"]["mean"] >= 0.10, ii_alpha_change_le_003=abs(vii["alpha"]["mean"]) <= 0.03)
                out[f"success_U5a_g{g}"]["all"] = all(out[f"success_U5a_g{g}"].values())
    return out


# ------------------------------------------------------------------ U5b
def u5b_kubric(split="test"):
    rows = [jl(f) for f in files(os.path.join(U5B, "kubric", split, "*.json"))]
    out = {"n": len(rows)}
    if not rows:
        return out
    per = {}
    for r in rows:
        R = r["rows"]; bD = R["base_D"]
        for vn in ("a", "b", "c", "d"):
            if vn not in R:
                continue
            ee = list(R[vn].values())
            per.setdefault(vn, []).append(dict(
                recovery=np.nanmean([x["recovery"] for x in ee]),
                one_minus_e=np.nanmean([1 - (0.5 * (x["rpe_r"] / bD["rpe_r"] + x["rpe_t"] / bD["rpe_t"])) for x in ee]),
                d_rpe_r=np.nanmean([x["rpe_r"] - bD["rpe_r"] for x in ee]),
                d_epe_moving=np.nanmean([x["epe_moving"] - bD["epe_moving"] for x in ee]),
                d_alpha=np.nanmean([x["alpha"] - bD["alpha"] for x in ee]) if bD.get("identifiable") else np.nan,
                d_epe_static=np.nanmean([x["epe_static"] - bD["epe_static"] for x in ee])))
    out["variants"] = {vn: {k: boot_mean([x[k] for x in v]) for k in v[0]} for vn, v in per.items()}
    out["contamination_D_minus_B_rpe_r"] = boot_mean([r["rows"]["base_D"]["rpe_r"] - r["rows"]["base_B"]["rpe_r"] for r in rows])
    out["probe_iou_gt_median"] = med([r["rows"]["probe_iou_gt"] for r in rows])
    out["probe_recall_gt_median"] = med([r["rows"]["probe_recall_gt"] for r in rows])
    out["mask_frac_median"] = med([r["rows"]["mask_frac"] for r in rows])
    if "a" in out["variants"] and "d" in out["variants"]:
        a, d = out["variants"]["a"], out["variants"]["d"]
        out["success_kubric"] = dict(a_reaches_half_oracle=bool(a["one_minus_e"]["mean"] >= 0.5 * d["one_minus_e"]["mean"]),
                                     a_moving_epe_le_1px=bool(a["d_epe_moving"]["mean"] <= 1.0))
    return out


def u5b_real():
    rows = [jl(f) for f in files(os.path.join(U5B, "real", "*.json")) if not os.path.basename(f).startswith("_")]
    out = {"n_windows": len(rows)}
    if not rows:
        return out
    d = [np.mean([x["rpe_r"] for x in r["c"].values()]) - r["a"]["r0"]["rpe_r"] for r in rows]
    out["T10"] = dict(boot_mean(d), definition="d_w = mean_r RPE-R(c_r) - RPE-R(a), pooled windows; one-sided p for mean d > 0")
    for st in sorted({r["set"] for r in rows}):
        rr = [r for r in rows if r["set"] == st]
        out[st] = {"n": len(rr), **{m: {"base": med([r["base"][m] for r in rr]), "a": med([r["a"]["r0"][m] for r in rr]),
                                        "b": med([r["b"]["r0"][m] for r in rr]),
                                        "c_mean": med([np.mean([x[m] for x in r["c"].values()]) for r in rr])}
                                    for m in ("ate", "rpe_t", "rpe_r")},
                   "d_rpe_r_c_minus_a": boot_mean([np.mean([x["rpe_r"] for x in r["c"].values()]) - r["a"]["r0"]["rpe_r"] for r in rr]),
                   "d_rpe_r_base_minus_a": boot_mean([r["base"]["rpe_r"] - r["a"]["r0"]["rpe_r"] for r in rr])}
    u2 = [f for f in glob.glob(os.path.join(RES, "u2", "**", "*"), recursive=True) if "more" in os.path.basename(f).lower()]
    out["more_variant_e"] = {"source": "behaviour track pose-set rows on the same windows (not run here)", "candidate_files": u2}
    return out


# ------------------------------------------------------------------ figures
def figures(fz, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (s, lab, key) in zip(axs, [("objall", "tau_O (objall)", "p_obj"), ("cam", "tau_C (cam)", "p_cam"), ("cont", "kappa (cont)", "p_cont")]):
        rk = jl(os.path.join(U4A, f"rank_{s}.json")); M = np.full((24, 12), np.nan)
        for n, v in rk["median"].items():
            if not n.endswith("r"):
                b, h = int(n[1:].split("h")[0]), int(n.split("h")[1]); M[b, h] = v
        im = ax.imshow(M, aspect="auto", cmap="viridis"); fig.colorbar(im, ax=ax)
        sel = fz["head_sets_used"][s]
        for n in sel:
            b, h = int(n[1:].split("h")[0]), int(n.split("h")[1]); ax.plot(h, b, "rs", mfc="none", ms=9)
        for n in FIVE:
            b, h = int(n[1:].split("h")[0]), int(n.split("h")[1]); ax.plot(h, b, "wx", ms=7)
        ax.set_title(f"{lab}, fit median; red = top-{fz['k_used'][s]}, x = E5 heads", fontsize=9); ax.set_xlabel("head"); ax.set_ylabel("block")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4a_pathway_maps_fit.png"), dpi=120); plt.close(fig)
    T = S["U4a"]["T9_test"]
    if T.get("status") == "ok":
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (k, xl, yl) in zip(axs, [("P_obj", "tau_O", "camera leakage lambda_O"), ("P_cam", "tau_C", "flow leakage lambda_C")]):
            d = T[k]
            ax.scatter(d["null_medians"], d["null_leakage_medians"], c="0.6", s=14, label="40 matched nulls")
            ax.scatter([d["observed_median"]], [d["leakage_median"]], c="r" if k == "P_obj" else "b", s=60, label=f"{k} (k={d['k']})")
            ax.axvline(d["null_pct95"], ls="--", c="0.4"); ax.axhline(0.2, ls=":", c="k")
            ax.set_xlabel(xl + " (test, scene median)"); ax.set_ylabel(yl); ax.legend(fontsize=8)
        fig.suptitle(f"F4b double dissociation (test): p = {T['p']:.3f}, passes = {T['passes']}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4b_double_dissociation_test.png"), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, c in zip(["objall", "cam", "cont", "obj"], ["r", "b", "g", "m"]):
        for split, ls in (("fit", "-"), ("test", "--"), ("transfer", ":")):
            cc = S["U4a"]["completeness"][s].get(split, {}).get("completeness")
            if cc:
                ax.plot(KGRID, [cc[str(k)] if str(k) in cc else cc[k] for k in KGRID], ls, c=c, marker="o", ms=3, label=f"{s} {split}")
    ax.axhline(0.8, c="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(KGRID); ax.set_xticklabels(KGRID)
    ax.set_xlabel("k"); ax.set_ylabel("completeness (scene median)"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4c_completeness.png"), dpi=120); plt.close(fig)
    U = S["U5a"].get("test", {}).get("variants", {})
    if U:
        vs = [v for v in ["i_g1.5", "ii_g1.5", "iii_g1.5", "iv_g1.5", "v", "i_g2", "ii_g2", "iii_g2", "iv_g2"] if v in U]
        fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(vs))
        for off, k, c in ((-0.2, "alpha", "r"), (0.2, "beta", "b")):
            m = [U[v][k]["mean"] for v in vs]; lo = [U[v][k]["mean"] - U[v][k]["lo"] for v in vs]; hi = [U[v][k]["hi"] - U[v][k]["mean"] for v in vs]
            ax.bar(x + off, m, 0.4, yerr=[lo, hi], color=c, label="d " + k)
        ax.axhline(0.10, ls=":", c="k"); ax.axhline(0, c="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(vs, rotation=30); ax.legend(); ax.set_title("F5a U5a edits, v2 test (identifiable)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "F5a_u5a_edits_test.png"), dpi=120); plt.close(fig)
    K = S["U5b"]["kubric_test"].get("variants", {}); Rr = S["U5b"]["real"]
    if K:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4)); vs = [v for v in "abcd" if v in K]
        axs[0].bar(vs, [K[v]["recovery"]["mean"] for v in vs], yerr=[[K[v]["recovery"]["mean"] - K[v]["recovery"]["lo"] for v in vs],
                                                                    [K[v]["recovery"]["hi"] - K[v]["recovery"]["mean"] for v in vs]])
        axs[0].set_title("U5b Kubric test D: recovery toward B"); axs[0].axhline(0, c="k", lw=0.6)
        sets = [s for s in ("sintel", "tum", "bonn") if s in Rr]
        for j, s in enumerate(sets):
            vals = [Rr[s]["rpe_r"][v] for v in ("base", "a", "b", "c_mean")]
            axs[1].bar(np.arange(4) + j * 0.25, vals, 0.25, label=s)
        axs[1].set_xticks(np.arange(4) + 0.25); axs[1].set_xticklabels(["base", "a", "b", "c (mean)"]); axs[1].legend()
        axs[1].set_title("U5b real pose sets: median RPE-R (deg)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "F5b_u5b_masking.png"), dpi=120); plt.close(fig)


def main():
    fz = jl(os.path.join(U4A, "fit_derived_values.json"))
    S = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "frozen_at": fz["frozen_at"], "kstar": {s: fz["kstar"][s]["kstar"] for s in fz["kstar"]}}
    S["U4a"] = {"T9_test": t9(fz, "test"), "T9_transfer_descriptive": t9(fz, "transfer"),
                "completeness": {s: {sp: completeness(s, sp) for sp in ("fit", "test", "transfer")} for s in ("objall", "cam", "cont", "obj")},
                "necessity_test": {s: necessity(s, "test") for s in ("objall", "cam", "cont", "obj")},
                "camwarp": {sp: camwarp(sp) for sp in ("test", "transfer")},
                "H3": {"fit_objall": fz["h3_five_head_rank_objall_fit"], "test_objall": h3_test("test")}}
    for s in ("cont", "obj"):
        S["U4a"][f"joint_{s}_test_k*"] = {}
        k = fz["k_used"][s]
        if k:
            met = {"cont": "kappa", "obj": "tau_O"}[s]
            side = t9_side(s, "test", k, met, "dpose_rel" if s == "obj" else "dF_rel")
            S["U4a"][f"joint_{s}_test_k*"] = {kk: v for kk, v in side.items() if kk not in ("null_medians", "null_leakage_medians", "null_medians_nb")}
    S["U5a"] = {ev: u5a_stats(ev) for ev in ("test", "transfer", "kitti15", "sintel")}
    S["U5b"] = {"kubric_test": u5b_kubric("test"), "real": u5b_real()}
    T9 = S["U4a"]["T9_test"]; T5 = S["U5a"]["test"]; T10 = S["U5b"]["real"].get("T10", {})
    conf = {"written": S["written"], "note": "raw one-sided p-values; Holm is applied by the lead over the full family",
            "T9": {"p": T9.get("p"), "status": T9.get("status"), "confirmatory": T9.get("confirmatory"), "note": T9.get("note"),
                   "passes_success_rule": T9.get("passes"), "k_used": [fz["k_used"]["objall"], fz["k_used"]["cam"]],
                   "n_scenes": [T9.get("P_obj", {}).get("n_scenes"), T9.get("P_cam", {}).get("n_scenes")]},
            "T5p": {"p": T5.get("T5p_g1.5", {}).get("p_gt0"), "mean_d": T5.get("T5p_g1.5", {}).get("mean"),
                    "ci": [T5.get("T5p_g1.5", {}).get("lo"), T5.get("T5p_g1.5", {}).get("hi")], "n": T5.get("T5p_g1.5", {}).get("n"),
                    "g2_secondary": T5.get("T5p_g2")},
            "T10": {"k_used": fz["u5b"]["k"], "confirmatory_as_specified": bool(fz["kstar"]["cont"]["kstar"]),
                    "note": None if fz["kstar"]["cont"]["kstar"] else fz["u5b"]["k_reason"], "p": T10.get("p_gt0"), "mean_d": T10.get("mean"), "ci": [T10.get("lo"), T10.get("hi")], "n_windows": T10.get("n")}}
    json.dump(conf, open(os.path.join(RES, "u4u5_confirmatory.json"), "w"), indent=1, default=float)
    json.dump(S, open(os.path.join(RES, "u4u5_summary.json"), "w"), indent=1, default=float)
    try:
        figures(fz, S)
    except Exception as e:  # figures are drafts; never block the statistics
        print("[summary] figure error", repr(e))
    print(json.dumps(conf, default=float)[:2000])


if __name__ == "__main__":
    main()
