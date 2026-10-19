#!/usr/bin/env python3
"""U4a interchange screens on Any4D (unified plan 5, U4a items 1 to 7). Uses any4d_hooks.py.

Scenes: MotionFactorial base scenes $E5_BIG/attr/factorial/<set>/<seed>/{A,B,C,D,D_gt}. Split rule (plan 2.1):
  fit = v2 even seeds, test = v2 odd seeds, transfer = v3float (all seeds). --split selects; --allow_test is required
  for test/transfer (thresholds and k* are pre-registered on Oct 27 before any test-split result).
Outputs (model scale, 7 views 0,4,8,12,16,20,23, target view 23):
  F(y)    scene flow of view-0 moving-object query points (gt_factorial q_pix, q_group > 0) to view 23, (Nm,3) flattened
  pose(y) views 1..6 relative to view 0: first two rotation columns (6D) and translation / median view-0 depth
  e(y)    0.5 * (RPE-R(y)/RPE-R(D) + RPE-T(y)/RPE-T(D)), pose.py (Sim(3) on centres; GT c2w = inv(w2c_cv)); raw RPEs kept
Definitions (per component k, y_k = base run with k patched):
  tau_O    = <F(y)-F(D), F(B)-F(D)> / |F(B)-F(D)|^2     lambda_O = |pose(y)-pose(D)| / |pose(C)-pose(D)|   (obj screen)
  tau_C    = <pose(y)-pose(D), pose(C)-pose(D)> / |pose(C)-pose(D)|^2   lambda_C = |F(y)-F(D)| / |F(B)-F(D)|   (cam screen)
  kappa(h) = (e(D) - e(D|h)) / (e(D) - e(B)); kappa_R, kappa_T per component too                          (cont screen)
Screens:
  obj   base D, source B, positions = view-0 tokens (pixel-aligned by the frame-0 copy)
  cam   base D, source C, positions = view-0 tokens + scale token
  camwarp (secondary) base D, source C at view-t tokens (t=1..6), C's token chosen by GT static-scene reprojection of the
        D token centre (D depth, D camera -> C camera), nearest token; floor = D->D through the round trip D->C->D map.
  objall (secondary) base D, source B at ALL view tokens (camera shared by B and D, so background tokens are aligned;
        object tokens carry the counterfactual). Needed because view-0 patches after block 21 cannot reach the view-23
        scene flow (it is decoded from view-6 tokens and block 23 is frame attention), so view-0 tau_O is 0 by
        construction for blocks 22 and 23 (E5 head b22h9).
  cont  base D, per head: block attention from queries of views 1..6 to keys on GT moving tokens (>= 25% moving
        pixels), renormalised; components = 288 heads.
  Every screen also runs ALL_HEADS (all 288 heads at once: completeness denominator) and records per-component base
  activation norm at the screen positions (for norm-matched nulls).
Modes:
  screen     --screen obj|cam|camwarp|cont [--chunk i/n] per scene -> results/u4a/screen_<screen>/<set>_<seed>[_c<i>of<n>].json
  rank       aggregate fit-scene screens -> results/u4a/rank_<screen>.json (median over scenes; E5 five-head ranks in tau_O)
  necessity  resample-ablate top-k components (all positions; donor = a random other scene of the same split)
  joint      top-k joint patch over HEADS (P_obj, P_cam, P_cont are head sets; residual states are reported as a layer
             profile only, because a residual patch carries everything upstream), k in {1,3,5,10,20}, plus --n_null
             matched nulls (same block, nearest activation norm) -> completeness = effect / effect of all 288 heads
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import any4d_hooks as ah  # noqa
import pose as pm  # noqa  (src/attr/pose.py)

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
FACT = os.path.join(E5_BIG, "attr", "factorial")
RES = os.environ.get("U4A_RES", os.path.join(E5_BIG, "results", "u4a"))
KGRID = [1, 3, 5, 10, 20]


# ------------------------------------------------------------------ scenes and splits
def scene_ok(d):
    return all(os.path.exists(os.path.join(d, v, "rgba_00023.png")) for v in "ABCD") and \
        all(os.path.exists(os.path.join(d, v, "gt_factorial.npz")) for v in ("A", "B", "C", "D_gt"))


def split_scenes(split, sets=None):
    out = []
    if split in ("fit", "test"):
        for d in sorted(glob.glob(os.path.join(FACT, "v2", "*"))):
            s = int(os.path.basename(d))
            if (s % 2 == 0) == (split == "fit") and scene_ok(d):
                out.append(d)
    elif split == "transfer":
        out = [d for d in sorted(glob.glob(os.path.join(FACT, "v3float", "*"))) if scene_ok(d)]
    elif split == "any":   # code validation only
        out = [d for d in sorted(glob.glob(os.path.join(FACT, "*", "*"))) if scene_ok(d)]
    return out


def sid(d):
    return f"{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}"


def gt_of(d, v):
    return np.load(os.path.join(d, "D_gt" if v == "D" else v, "gt_factorial.npz"))


def to518(xy, W=512):
    s = 518 / W
    return np.clip(np.round((np.asarray(xy) + 0.5) * s - 0.5).astype(int), 0, 517)


# ------------------------------------------------------------------ output functionals
class Readout:
    def __init__(self, d):
        g = gt_of(d, "D")
        q = g["q_pix"]; grp = g["q_group"]
        self.mv = grp > 0
        uv = to518(q)
        self.u = torch.as_tensor(uv[self.mv, 0], device="cuda"); self.v = torch.as_tensor(uv[self.mv, 1], device="cuda")
        self.gt_c2w = {}
        for vn in "ABCD":
            gg = gt_of(d, vn)
            fr = list(gg["frames"]) if "frames" in gg.files else list(range(24))
            idx = [fr.index(f) if f in fr else f for f in ah.FRAMES]
            self.gt_c2w[vn] = np.linalg.inv(gg["w2c_cv"][idx])
        self.moving_tok = {vn: ah.token_mask_from_pixels(gt_of(d, vn)["moving_mask"][ah.FRAMES], 0.25).reshape(-1).cuda()
                           for vn in "ABCD"}
        self.moving_tok_label = {vn: ah.token_mask_from_pixels(gt_of(d, vn)["moving_mask"][ah.FRAMES], 0.5).reshape(-1)
                                 for vn in "ABCD"}

    def F(self, out):
        return out["scene_flow"][-1][self.v, self.u].reshape(-1)

    @staticmethod
    def pose(out):
        P = out["cam_pose"]; rel = torch.linalg.inv(P[0])[None] @ P[1:]
        z = ((out["pts3d_0"] - P[0, :3, 3]) @ P[0, :3, :3])[..., 2]
        md = z.median().clamp_min(1e-6)
        return torch.cat([rel[:, :3, :2].reshape(-1), (rel[:, :3, 3] / md).reshape(-1)])

    def rpe(self, out, vn="D"):
        t, r = pm.rpe(out["cam_pose"].double().cpu().numpy(), self.gt_c2w[vn])
        return t, r


def dot(a, b):
    return float((a * b).sum())


class SceneRuns:
    """Base runs and caches for one base scene."""
    def __init__(self, H, d, need=("D", "B", "C"), cache=True):
        self.H, self.d = H, d
        self.ro = Readout(d)
        self.views = {v: ah.load_views(os.path.join(d, v)) for v in need}
        self.V = len(self.views["D"])
        self.out, self.cache = {}, {}
        for v in need:
            o, cch = H.run(self.views[v], capture=cache)
            self.out[v] = o; self.cache[v] = cch
        o = self.out
        self.FD, self.FB = self.ro.F(o["D"]), self.ro.F(o["B"])
        self.pD = self.ro.pose(o["D"])
        self.pC = self.ro.pose(o["C"]) if "C" in o else None
        self.dFB = self.FB - self.FD; self.nFB = float((self.dFB ** 2).sum())
        if self.pC is not None:
            self.dPC = self.pC - self.pD; self.nPC = float((self.dPC ** 2).sum())
        self.rpeD = self.ro.rpe(o["D"], "D"); self.rpeB = self.ro.rpe(o["B"], "D")

    def e(self, rpe):
        return 0.5 * (rpe[1] / max(self.rpeD[1], 1e-9) + rpe[0] / max(self.rpeD[0], 1e-9))

    def measures(self, y):
        dF = self.ro.F(y) - self.FD; dP = self.ro.pose(y) - self.pD
        m = {"tau_O": dot(dF, self.dFB) / max(self.nFB, 1e-12),
             "dF_rel": float(dF.norm()) / max(self.nFB ** 0.5, 1e-12)}
        if self.pC is not None:
            m["tau_C"] = dot(dP, self.dPC) / max(self.nPC, 1e-12)
            m["dpose_rel"] = float(dP.norm()) / max(self.nPC ** 0.5, 1e-12)
        r = self.ro.rpe(y, "D"); eD = 1.0; eB = self.e(self.rpeB); ey = self.e(r)
        m.update(rpe_t=r[0], rpe_r=r[1], e=ey, kappa=(eD - ey) / (eD - eB) if abs(eD - eB) > 1e-9 else float("nan"),
                 kappa_R=(self.rpeD[1] - r[1]) / (self.rpeD[1] - self.rpeB[1]) if abs(self.rpeD[1] - self.rpeB[1]) > 1e-9 else float("nan"),
                 kappa_T=(self.rpeD[0] - r[0]) / (self.rpeD[0] - self.rpeB[0]) if abs(self.rpeD[0] - self.rpeB[0]) > 1e-9 else float("nan"))
        return m

    def summary(self):
        return {"nFB": self.nFB, "nPC": getattr(self, "nPC", None), "rpe_D": self.rpeD, "rpe_B": self.rpeB,
                "e_B": self.e(self.rpeB), "n_moving_queries": int(self.ro.mv.sum()),
                "n_moving_tokens_D": int(self.ro.moving_tok["D"].sum())}


# ------------------------------------------------------------------ warp map (secondary camera analysis)
def warp_map(d, src_v="C", dst_v="D", V=7):
    """For every D token of views 1..V-1: the C token (same view) hit by GT static-scene reprojection of the token centre.
    Returns (dst_flat, src_flat) long tensors and the round-trip map D->C->D for the misregistration floor."""
    sys.path.insert(0, os.path.join(os.environ.get("E5_REPO", "/data2/DATA/rohit_sinha/e5/repo"), "src"))
    import e5_kubric_post as kp  # noqa
    mD, gD, _ = kp.load_scene(os.path.join(d, dst_v)); mC, gC, _ = kp.load_scene(os.path.join(d, src_v))
    W = mD["resolution"][0]; s = W / 518
    cen = (np.arange(37) * 14 + 7) * s - 0.5    # token centre in 512 px
    uu, vv = np.meshgrid(cen, cen)
    ui, vi = np.clip(np.round(uu).astype(int), 0, W - 1), np.clip(np.round(vv).astype(int), 0, W - 1)

    def one(ma, ga, mb, f):
        X = kp.backproject(ma, ga["depth"][f, ..., 0], f)[vi, ui]
        xb = kp.project(mb, X, f)[0]
        tu = np.floor((xb[..., 0] + 0.5) / s / 14).astype(int); tv = np.floor((xb[..., 1] + 0.5) / s / 14).astype(int)
        ok = (tu >= 0) & (tu < 37) & (tv >= 0) & (tv < 37)
        return (tv * 37 + tu).reshape(-1), ok.reshape(-1)

    dst, src, rt = [], [], []
    for vi_, f in enumerate(ah.FRAMES):
        if vi_ == 0:
            continue
        m1, ok1 = one(mD, gD, mC, f); m2, ok2 = one(mC, gC, mD, f)
        idx = np.nonzero(ok1)[0]
        ok_rt = ok2[m1[idx]]
        base = vi_ * ah.P_TOK
        dst.append(base + idx); src.append(base + m1[idx])
        rt.append((base + idx[ok_rt], base + m2[m1[idx[ok_rt]]]))
    T = lambda a: torch.as_tensor(np.concatenate(a), device="cuda")
    return T(dst), T(src), T([r[0] for r in rt]), T([r[1] for r in rt])


# ------------------------------------------------------------------ screens
def comps_for(screen, resid=True):
    if screen == "cont":
        return ah.all_components(resid=False)
    return ah.all_components(resid=resid)


def act_norms(cache, comps, pos):
    out = {}
    for c in comps:
        x = cache[(c[0], c[1])]
        p = pos[pos < x.shape[0]]
        sl = slice(c[2] * 64, (c[2] + 1) * 64) if c[0] == "h" else slice(None)
        out[ah.comp_name(c)] = float(x[p][:, sl].float().norm(dim=-1).mean())
    return out


def run_screen(H, d, screen, chunk=(0, 1), comps=None):
    t0 = time.time()
    S = SceneRuns(H, d)
    V = S.V
    comps = comps or comps_for(screen)
    comps = comps[chunk[0]::chunk[1]]
    rows = {}
    if screen in ("obj", "cam", "objall"):
        pos = ah.view_positions([0], V, scale=(screen == "cam")) if screen != "objall" else ah.view_positions(range(V), V)
        src = S.cache["C" if screen == "cam" else "B"]
        norms = act_norms(S.cache["D"], comps, pos)
        for c in comps:
            y, _ = H.run(S.views["D"], patches=[ah.Patch(c, pos, src)])
            rows[ah.comp_name(c)] = S.measures(y)
        y, _ = H.run(S.views["D"], patches=[ah.Patch(c, pos, src) for c in ah.all_components(resid=False)])
        rows["ALL_HEADS"] = S.measures(y)
        y, _ = H.run(S.views["D"], patches=[ah.Patch(("r", 23, None), pos, src)])
        rows["FINAL_RESID"] = S.measures(y)
    elif screen == "camwarp":
        dst, srcp, rtd, rts = warp_map(d, V=V)
        norms = act_norms(S.cache["D"], comps, dst)
        for c in comps:
            y, _ = H.run(S.views["D"], patches=[ah.Patch(c, dst, S.cache["C"], srcp)])
            f, _ = H.run(S.views["D"], patches=[ah.Patch(c, rtd, S.cache["D"], rts)])
            m = S.measures(y); fl = S.measures(f)
            m.update({f"floor_{k}": v for k, v in fl.items()})
            rows[ah.comp_name(c)] = m
    elif screen == "cont":
        km = S.ro.moving_tok["D"]
        norms = act_norms(S.cache["D"], comps, ah.view_positions(range(1, V), V))
        for c in comps:
            y, _ = H.run(S.views["D"], masks={c[1]: ah.KeyMask([c[2]], km)})
            rows[ah.comp_name(c)] = S.measures(y)
        y, _ = H.run(S.views["D"], masks={b: ah.KeyMask(list(range(12)), km) for b in range(24)})
        rows["ALL_HEADS"] = S.measures(y)
    base = {v: S.measures(S.out[v]) for v in S.out}
    return {"scene": sid(d), "screen": screen, "chunk": list(chunk), "summary": S.summary(), "base": base,
            "rows": rows, "norms": norms, "wall_s": round(time.time() - t0, 1),
            "s_per_forward": round((time.time() - t0) / max(1, len(rows) + 3), 3)}


# ------------------------------------------------------------------ ranking, nulls, joint
METRIC = {"obj": "tau_O", "cam": "tau_C", "cont": "kappa", "camwarp": "tau_C", "objall": "tau_O"}


def load_screen(screen, scenes):
    per = {}
    for d in scenes:
        for f in glob.glob(os.path.join(RES, f"screen_{screen}", f"{sid(d)}*.json")):
            r = json.load(open(f)); e = per.setdefault(sid(d), {"rows": {}, "norms": {}, "summary": r["summary"]})
            e["rows"].update(r["rows"]); e["norms"].update(r["norms"])
    return per


def rank(screen, scenes):
    per = load_screen(screen, scenes); met = METRIC[screen]
    names = sorted({k for e in per.values() for k in e["rows"] if k not in ("ALL_HEADS", "FINAL_RESID")})
    if screen == "cont":   # pooled kappa: sum_s (e_D - e_y) / sum_s (e_D - e_B), e_D = 1 (per-scene ratios are unstable)
        den = sum(1.0 - e["summary"]["e_B"] for e in per.values())
        med = {n: float(sum(1.0 - e["rows"][n]["e"] for e in per.values() if n in e["rows"]) / den) if abs(den) > 1e-9 else float("nan")
               for n in names}
    else:
        med = {n: float(np.nanmedian([e["rows"][n][met] for e in per.values() if n in e["rows"]])) for n in names}
    order = sorted(names, key=lambda n: -med[n])
    out = {"screen": screen, "metric": met, "n_scenes": len(per), "order": order, "median": med,
           "norm": {n: float(np.nanmedian([e["norms"][n] for e in per.values() if n in e["norms"]])) for n in names},
           "all_heads_median": float(np.nanmedian([e["rows"]["ALL_HEADS"][met] for e in per.values() if "ALL_HEADS" in e["rows"]]))
           if per else float("nan")}
    heads_only = [n for n in order if "h" in n[1:]]
    out["five_rank_heads"] = {f"b{b}h{h}": (heads_only.index(f"b{b}h{h}") + 1 if f"b{b}h{h}" in heads_only else None)
                              for b, h in ah.FIVE}
    out["H3_all_five_top20"] = all(v is not None and v <= 20 for v in out["five_rank_heads"].values())
    return out


def parse(n):
    b = int(n[1:].split("h")[0].split("r")[0])
    return ("r", b, None) if n.endswith("r") else ("h", b, int(n.split("h")[1]))


def matched_null(top, norms, rng, all_names):
    """One null set: per component, a different component of the same block and type (head vs residual; frame vs global
    is implied by the block for heads), drawn among the 6 nearest in activation norm. Residual states are matched by
    block parity (global/frame) and nearest norm."""
    used = set(top); pick = []
    all_names = sorted(set(all_names) | {ah.comp_name(c) for c in ah.all_components()})
    if all(not n.endswith("r") for n in top):
        all_names = [m for m in all_names if not m.endswith("r")]
    for n in top:
        c = parse(n)
        if c[0] == "h":
            pool = [m for m in all_names if m.startswith(f"b{c[1]}h") and m not in used]
        else:
            pool = [m for m in all_names if m.endswith("r") and parse(m)[1] % 2 == c[1] % 2 and m not in used]
        if not pool:
            pool = [m for m in all_names if m.endswith("r") == n.endswith("r") and m not in used]
        pool = sorted(pool, key=lambda m: abs(norms.get(m, 0) - norms.get(n, 0)))[:6]
        m = pool[int(rng.integers(len(pool)))]; used.add(m); pick.append(m)
    return pick


def patch_set(S, screen, names, src_scene=None):
    V = S.V
    if screen == "cont":
        km = S.ro.moving_tok["D"]; masks = {}
        for n in names:
            c = parse(n); masks.setdefault(c[1], ah.KeyMask([], km)).heads.append(c[2])
        return dict(masks=masks)
    pos = ah.view_positions([0], V, scale=(screen == "cam")) if src_scene is None else torch.arange(V * ah.P_TOK + 1, device="cuda")
    src = src_scene if src_scene is not None else S.cache["B" if screen == "obj" else "C"]
    return dict(patches=[ah.Patch(parse(n), pos, src) for n in names])


def run_joint(H, d, screen, rk, n_null=40, kgrid=KGRID, seed=0):
    S = SceneRuns(H, d); rng = np.random.default_rng(seed)
    met = METRIC[screen]; rows = {}
    names = [n for n in rk["order"] if not n.endswith("r")]   # P_* are head sets; residual states = layer profile only
    y, _ = H.run(S.views["D"], **patch_set(S, screen, [ah.comp_name(c) for c in ah.all_components(resid=False)]))
    full = S.measures(y); rows["ALL_HEADS"] = full
    for k in kgrid:
        top = names[:k]
        y, _ = H.run(S.views["D"], **patch_set(S, screen, top)); m = S.measures(y)
        m["completeness"] = m[met] / full[met] if abs(full[met]) > 1e-12 else float("nan")
        nulls = []
        for r in range(n_null):
            ns = matched_null(top, rk["norm"], rng, names)
            yn, _ = H.run(S.views["D"], **patch_set(S, screen, ns)); nulls.append(S.measures(yn))
        if nulls:
            nv = np.array([x[met] for x in nulls])
            m["null_values"] = nv.tolist(); m["null_pct95"] = float(np.percentile(nv, 95))
            m["rank_in_null"] = float((nv >= m[met]).mean())
        rows[f"k{k}"] = m
    return {"scene": sid(d), "screen": screen, "summary": S.summary(), "rows": rows}


def run_necessity(H, d, donor, screen, rk, kgrid=KGRID):
    S = SceneRuns(H, d)
    _, dc = H.run(ah.load_views(os.path.join(donor, "D")), capture=True)
    rows = {}
    for k in kgrid:
        heads = [n for n in rk["order"] if not n.endswith("r")][:k]
        y, _ = H.run(S.views["D"], **patch_set(S, screen if screen != "cont" else "obj", heads, src_scene=dc))
        rows[f"k{k}"] = S.measures(y)
    return {"scene": sid(d), "donor": sid(donor), "screen": screen, "rows": rows}


def kstar(screen, scenes, tag="", frac=0.8):
    """k* = smallest k in KGRID whose scene-median completeness (joint top-k effect / ALL_HEADS effect) reaches 80%,
    from joint runs on FIT scenes (results/u4a/joint_<screen>/). Writes results/u4a/kstar.json (key = screen)."""
    rows = [json.load(open(f)) for d in scenes for f in glob.glob(os.path.join(RES, f"joint_{screen}{tag}", sid(d) + ".json"))]
    curve = {k: float(np.nanmedian([r["rows"][f"k{k}"]["completeness"] for r in rows])) if rows else float("nan") for k in KGRID}
    ks = [k for k in KGRID if curve[k] >= frac]
    out = {"screen": screen, "n_scenes": len(rows), "completeness_median": curve, "kstar": ks[0] if ks else None,
           "rule": "smallest k in {1,3,5,10,20} with scene-median completeness >= 0.8 on fit scenes; None = distributed (risk 4)"}
    f = os.path.join(RES, "kstar.json"); allk = json.load(open(f)) if os.path.exists(f) else {}
    allk[screen] = out; json.dump(allk, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["screen", "rank", "joint", "necessity", "kstar"])
    p.add_argument("--screen", default="obj", choices=["obj", "cam", "camwarp", "cont", "objall"])
    p.add_argument("--split", default="fit", choices=["fit", "test", "transfer", "any"])
    p.add_argument("--allow_test", action="store_true")
    p.add_argument("--scenes", default="", help="comma list of <set>_<seed> to restrict to")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", default="0/1")
    p.add_argument("--comps", default="", help="comma list of component names (smoke)")
    p.add_argument("--n_null", type=int, default=40)
    p.add_argument("--rank_from", default="", help="rank json (default results/u4a/rank_<screen>.json)")
    p.add_argument("--out_tag", default="")
    a = p.parse_args()
    if a.split in ("test", "transfer") and a.mode != "rank" and not a.allow_test:
        sys.exit("test/transfer split requires --allow_test (after the Oct 27 pre-registration)")
    scenes = split_scenes(a.split)
    if a.scenes:
        want = set(a.scenes.split(",")); scenes = [d for d in scenes if sid(d) in want]
    if a.limit:
        scenes = scenes[:a.limit]
    print(f"[interchange] mode={a.mode} screen={a.screen} split={a.split} scenes={len(scenes)}", flush=True)
    if a.mode == "kstar":
        r = kstar(a.screen, scenes, a.out_tag); print(json.dumps(r)); return
    if a.mode == "rank":
        r = rank(a.screen, scenes); os.makedirs(RES, exist_ok=True)
        f = os.path.join(RES, f"rank_{a.screen}{a.out_tag}.json"); json.dump(r, open(f, "w"), indent=1); print(f, r["five_rank_heads"]); return
    model = ah.load_any4d(); H = ah.Any4DHooks(model)
    ci, cn = (int(x) for x in a.chunk.split("/"))
    comps = [parse(n) for n in a.comps.split(",")] if a.comps else None
    od = os.path.join(RES, f"{a.mode}_{a.screen}{a.out_tag}"); os.makedirs(od, exist_ok=True)
    rk = json.load(open(a.rank_from or os.path.join(RES, f"rank_{a.screen}.json"))) if a.mode in ("joint", "necessity") else None
    drng = np.random.default_rng(0)
    for i, d in enumerate(scenes):
        f = os.path.join(od, f"{sid(d)}" + (f"_c{ci}of{cn}" if cn > 1 else "") + ".json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        if a.mode == "screen":
            r = run_screen(H, d, a.screen, (ci, cn), comps)
        elif a.mode == "joint":
            r = run_joint(H, d, a.screen, rk, a.n_null)
        else:
            others = [x for x in split_scenes(a.split) if x != d]
            r = run_necessity(H, d, others[int(drng.integers(len(others)))], a.screen, rk)
        r["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        json.dump(r, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        print(f"[interchange] {sid(d)} {a.mode}/{a.screen} {round(time.time()-t0,1)} s peak {r['peak_gpu_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
