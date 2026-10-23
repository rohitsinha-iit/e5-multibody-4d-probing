#!/usr/bin/env python3
"""Freeze the fit-derived values (pre-registration block 'fit_derived_values'), from FIT-scene outputs only.
  build    -> $E5_BIG/results/u4a/fit_derived_values.json  (+ results/u5b/u5b_heads.json)
  append   -> replace the placeholder string of 'fit_derived_values' in the given pre-registration file(s) with the
              block; correct 'written' and add 'written_correction_note'; every other byte unchanged.
"""
import glob, hashlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
RES = os.path.join(E5_BIG, "results")
U4A, U5A, U5B = (os.path.join(RES, x) for x in ("u4a", "u5a", "u5b"))
FIVE = ["b2h1", "b22h9", "b17h10", "b14h6", "b8h11"]
SCREENS = ["objall", "cam", "cont", "obj"]
MET = {"obj": "tau_O", "objall": "tau_O", "cam": "tau_C", "cont": "kappa"}


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nanmed(x):
    x = np.asarray([v for v in x if v is not None], float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def necessity_summary(s):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(U4A, f"necessity_{s}", "*.json")))]
    out = {"n_scenes": len(rows)}
    for k in (1, 3, 5, 10, 20):
        r = [x["rows"][f"k{k}"] for x in rows if f"k{k}" in x["rows"]]
        out[f"k{k}"] = {m: nanmed([y.get(m) for y in r]) for m in ("dF_rel", "dpose_rel", "e", "rpe_r", "rpe_t")}
    return out


def build():
    ks = json.load(open(os.path.join(U4A, "kstar.json")))
    ranks = {s: json.load(open(os.path.join(U4A, f"rank_{s}.json"))) for s in SCREENS}
    heads = {s: [n for n in ranks[s]["order"] if not n.endswith("r")] for s in SCREENS}
    blk = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z (%z)"),
           "source": "fit split only (v2 even seeds, 50 base scenes); written by src/attr/freeze_fit.py before any "
                     "test-split or transfer-split mechanism run",
           "kstar": {}, "rank_order_heads_top20": {s: heads[s][:20] for s in SCREENS}}
    for s in SCREENS:
        k = ks[s]
        blk["kstar"][s] = {kk: k.get(kk) for kk in ("kstar", "n_scenes", "completeness_median", "completeness_pooled",
                                                      "kstar_nb", "completeness_median_nb", "topk_heads") if kk in k}
    kk = {s: blk["kstar"][s]["kstar"] for s in SCREENS}
    # k used by the test runs: k* when it exists; otherwise (k* = None, distributed pathway, plan risk 4) the largest
    # grid value k = 20, and every statistic that depends on it is descriptive (decided here, before any test run)
    blk["k_used"] = {s: (kk[s] if kk[s] else 20) for s in SCREENS}
    blk["k_used_reason"] = {s: ("k*" if kk[s] else "k* is None (no k in the grid reaches 80% scene-median completeness on "
                                "fit): fallback k = 20, statistics at this k are descriptive (plan section 11, risk 4)")
                            for s in SCREENS}
    blk["p_obj"] = heads["objall"][:kk["objall"]] if kk["objall"] else None
    blk["p_cam"] = heads["cam"][:kk["cam"]] if kk["cam"] else None
    blk["p_cont"] = heads["cont"][:kk["cont"]] if kk["cont"] else None
    blk["p_obj_view0_secondary"] = heads["obj"][:kk["obj"]] if kk["obj"] else None
    blk["head_sets_used"] = {s: heads[s][:blk["k_used"][s]] for s in SCREENS}
    blk["h3_five_head_rank_objall_fit"] = ranks["objall"]["five_rank_heads"]
    blk["h3_all_five_top20_fit"] = ranks["objall"]["H3_all_five_top20"]
    blk["necessity_fit"] = {s: necessity_summary(s) for s in SCREENS}
    pj = json.load(open(os.path.join(U5B, "probe_any4d.json")))
    blk["probe"] = {"L_star": pj["L_star"], "threshold_logit": pj["threshold"], "recall_at_thr_oof": pj["recall_at_thr_oof"],
                    "fpr_at_thr_oof": pj["fpr_at_thr_oof"], "cv_auc_by_layer": {str(L): v["cv_auc"] for L, v in pj["layers"].items()},
                    "n_scenes": pj["n_scenes"], "file": os.path.join(U5B, "probe_any4d.pt"),
                    "rule": "L* = best 5-fold (grouped by scene) held-out AUC over L in {13,15,17,19,21}; threshold = "
                            "10th percentile of out-of-fold positive scores at L* (90% recall); token labels >= 50% moving pixels"}
    import repair as rp
    dsum = json.load(open(os.path.join(U5A, "u5a_dirs_summary.json")))
    p_cont = [[int(n[1:].split("h")[0]), int(n.split("h")[1])] for n in blk["head_sets_used"]["cont"]]
    rsets = rp.random_head_sets(p_cont) if p_cont else []
    u5b_file = os.path.join(U5B, "u5b_heads.json")
    json.dump({"p_cont": p_cont, "random_head_sets": rsets, "rule": rp.random_head_sets.__doc__.strip()},
              open(u5b_file, "w"), indent=1)
    blk["u5a"] = {"heads": sorted(dsum), "cos_OC": {h: v["cos_OC"] for h, v in dsum.items()},
                  "projector_file": os.path.join(U5A, "u5a_dirs.pt"),
                  "random_projector_file": os.path.join(U5A, "u5a_dirs_random_projectors.pt"),
                  "g": [1.5, 2.0], "g_primary": 1.5, "n_random": 20,
                  "random_rule": rp.random_projectors.__doc__.strip(),
                  "centring": "m = mean head output over static tokens (all 7 views, GT label < 50% moving) of D, averaged "
                              "over fit scenes (per head); stored as key 'm' in projector_file",
                  "variant_v_checkpoint": rp.LORA["heads_sel"]}
    blk["u5b"] = {"p_cont": p_cont, "k": blk["k_used"]["cont"], "k_reason": blk["k_used_reason"]["cont"],
                  "random_head_sets": rsets, "file": u5b_file, "probe_file": blk["probe"]["file"]}
    blk["analysis_conventions"] = {
        "fallback_status": "T9 is confirmatory only if k* exists for both objall and cam; T10 is run exactly as specified "
                           "only if k* exists for cont. Otherwise the same computation is run at k_used and reported as "
                           "descriptive unless the lead decides otherwise (plan risk 4)",
        "T9": "test split (v2 odd seeds) primary, transfer descriptive. For P_obj (objall, tau_O) and P_cam (cam, tau_C) at "
              "k*: statistic = scene median over test scenes; null draw r (r = 1..40) uses the same matched component "
              "set on every scene (numpy default_rng(k) per scene), its statistic = scene median of that set; null rank "
              "p = (1 + #{null >= observed}) / 41; pass = observed above the 95th percentile of the 40 null medians for "
              "both, and scene-median leakage (lambda_O = dpose_rel of the P_obj patch, lambda_C = dF_rel of the P_cam "
              "patch) <= 0.2 for both; p_T9 = max(p_O, p_C). Boundary rule applied to F (primary); no-rule values reported",
        "T5p": "identifiable test scenes in the unedited D run (boundary rule). d_s = (alpha_i - alpha_base) - mean over the "
               "20 random projectors of (alpha_iii - alpha_base); H1 mean d > 0; one-sided scene bootstrap (2000, seed 0), "
               "p = (1 + #{bootstrap mean <= 0}) / 2001; g = 1.5 confirmatory, g = 2 secondary",
        "T10": "windows of results/u2/pose_windows.json pooled over Sintel, TUM-dynamics and Bonn; d_w = mean over the 20 "
               "frozen random head sets of RPE-R(c) - RPE-R(a); H1 mean d > 0; one-sided window bootstrap (2000, seed 0), "
               "p = (1 + #{bootstrap mean <= 0}) / 2001; per-set values descriptive",
        "U5a_success": "on identifiable test scenes (g = 1.5): mean d alpha >= 0.10, |mean d beta| <= 0.03, mean d moving "
                       "EPE <= +1 px, mean static EPE ratio - 1 <= 0.05 for (i); mean d beta >= 0.10 and |mean d alpha| <= 0.03 for (ii)",
        "U5b_success": "T10 significant after the lead's Holm correction; Kubric test D: mean (1 - e_a) >= 0.5 mean (1 - e_d) "
                       "with e normalised by the unedited D run; mean d moving EPE of (a) <= +1 px. The contamination-null check (D - B "
                       "RPE-R CI includes 0) uses the behaviour track's T8; this track also reports the Kubric test D - B "
                       "RPE-R CI from its own unedited runs (descriptive)",
        "component_ranking_note": "rank orders come from the fit screens, which used all moving query points (screens ran on "
                                  "2026-10-20 before the boundary rule was implemented in the readout); joint, k*, necessity "
                                  "and every test metric apply the boundary rule (primary) and also store the no-rule value"}
    files = [os.path.join(U4A, f"rank_{s}.json") for s in SCREENS] + [os.path.join(U4A, "kstar.json"),
             blk["probe"]["file"], os.path.join(U5B, "probe_any4d.json"), blk["u5a"]["projector_file"],
             blk["u5a"]["random_projector_file"], os.path.join(U5A, "u5a_dirs_summary.json"), u5b_file]
    blk["sha256"] = {f: sha(f) for f in files}
    out = os.path.join(U4A, "fit_derived_values.json")
    json.dump(blk, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    print("[freeze] wrote", out, "k*:", kk)
    return blk


PLACEHOLDER_KEY = '"fit_derived_values": "TO BE APPENDED'
OLD_WRITTEN = '"written": "2026-10-21 (IST), before any test-split or transfer-split result of U2, U4, U5"'
NEW_WRITTEN = ('"written": "2026-10-23 (IST); core committed in 678689f",{nl}'
               ' "written_correction_note": "the core said 2026-10-21 in error; git commit time is authoritative"')


def append(paths):
    blk = json.load(open(os.path.join(U4A, "fit_derived_values.json")))
    blk["sha256"][os.path.join(U4A, "fit_derived_values.json")] = sha(os.path.join(U4A, "fit_derived_values.json"))
    for p in paths:
        t = open(p, "rb").read().decode("utf-8")
        nl = "\r\n" if "\r\n" in t else "\n"
        assert t.count(OLD_WRITTEN) == 1 and t.count(PLACEHOLDER_KEY) == 1, p
        i = t.index(PLACEHOLDER_KEY); j = t.index('"', i + len('"fit_derived_values": "')) + 1
        body = json.dumps(blk, indent=1).replace("\n", nl + " ")
        t2 = t[:i] + '"fit_derived_values": ' + body + t[j:]
        t2 = t2.replace(OLD_WRITTEN, NEW_WRITTEN.format(nl=nl))
        json.loads(t2)
        open(p + ".tmp", "wb").write(t2.encode("utf-8")); os.replace(p + ".tmp", p)
        print("[freeze] appended to", p, sha(p))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "append":
        append(sys.argv[2:])
    else:
        build()
