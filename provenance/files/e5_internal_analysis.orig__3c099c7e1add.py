"""E5 Steps 4-6 on Any4D internals.  Usage: python e5_internal_analysis.py <internals_dir> <gt_dir> <out_prefix>

Patch set: view-0 patches with purity >= 0.8 and a visible GT correspondence in frame 23.
Moving-distinguishable set M: moving patches whose true and static-hypothesis locations in frame 23 differ by > 2 patches.

E5-A (attention, 12 global blocks x 12 heads, view-0 queries vs view-6 = frame 23 keys)
  static_hit / track_hit / rigid_hit : argmax lands within 1 patch (Chebyshev) of the GT match (static patches),
                                       the true moved location (M), or the static-hypothesis location (M).
  multibody gain (soft-argmax correspondences of a head chosen by static_hit on the other half of the scenes):
      gain_k = median Sampson(moving body k | F_bg fit on static patches)
               / median Sampson(held-out half of body k | F_k fit on the other half)    (cross-fitted, so 1 = no structure)
      computed for attention, Any4D outputs and GT.
E5-F (training-free clustering, all 25 residual-stream layers; features = view-0 token, view-6 token, concat)
  k-means (k = number of visible bodies) on PCA-32 of standardised tokens, balanced per instance.
  ARI vs rigid-body labels and vs instance labels; confound_same: share of cross-copy pairs split;
  confound_glued: share of cross-object pairs joined.
E5-P (trained probes, 5-fold cross-validation grouped by scene, features = concat(view-0, view-6) token)
  pair probe  : low-rank bilinear same-rigid-body classifier; AUC on held-out pairs, confound pair accuracies,
                K estimate from spectral eigengap of the probe affinity.
  motion probe: linear moving-vs-static AUC (per split).
  object-motion probe: ridge regression of d_obj = x2_true - x2_static (px); R^2 and slope (fraction recovered)
                on moving patches. Output-level slope from Any4D predictions for comparison.
"""
import glob, json, os, sys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_multibody as mb

G = 37; T = G * G; PATCH = 14
idir, gdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(0)
QUICK = os.environ.get("E5_QUICK") == "1"
LAYERS = [0, 12, 24] if QUICK else list(range(25))
if os.environ.get("E5_LAYERS"):
    LAYERS = [int(x) for x in os.environ["E5_LAYERS"].split(",")]
PARTS = set(os.environ.get("E5_PARTS", "A,F,P").split(","))
STEPS = 40 if QUICK else 300
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(idir, "*_tok.npy")))
GT = {n: dict(np.load(os.path.join(gdir, f"{n}_gt.npz"), allow_pickle=True)) for n in names}
for n in names:
    g = GT[n]; g["split"] = str(g["split"])
    g["pure"] = (g["purity"] >= 0.8) & g["vis"]
    dist = np.linalg.norm(g["x2_true"] - g["x2_static"], axis=1)
    g["M"] = g["pure"] & (g["body"] > 0) & (dist > 2 * PATCH)
    g["S"] = g["pure"] & (g["body"] == 0)
seed_of = {n: int(n.split("_s")[-1]) for n in names}
fold2 = {n: seed_of[n] % 2 for n in names}
print("scenes", len(names), flush=True)


def cheb_hit(idx_or_xy, target_xy):
    if idx_or_xy.ndim == 1:
        cx, cy = idx_or_xy % G, idx_or_xy // G
    else:
        cx, cy = idx_or_xy[:, 0], idx_or_xy[:, 1]
    tx, ty = (target_xy[:, 0] - 6.5) / PATCH, (target_xy[:, 1] - 6.5) / PATCH
    return (np.abs(cx - tx) <= 1.0) & (np.abs(cy - ty) <= 1.0)


def fit_F(x1, x2):
    a, Ta = mb.hartley_normalize(x1); b, Tb = mb.hartley_normalize(x2)
    return Tb.T @ mb.eight_point(a, b) @ Ta


def samp(F, x1, x2):
    return np.sqrt(mb.sampson(F, np.c_[x1, np.ones(len(x1))], np.c_[x2, np.ones(len(x2))]))


def multibody_gain(x1, x2, body, S):
    """Cross-fitted gain per moving body; returns list of log10 gains."""
    if S.sum() < 12:
        return []
    Fbg = fit_F(x1[S], x2[S])
    out = []
    for k in sorted(set(body[body > 0])):
        idx = np.where(body == k)[0]
        if len(idx) < 24:
            continue
        perm = rng.permutation(idx); h = len(perm) // 2; A, B = perm[:h], perm[h:]
        own = 0.5 * (np.median(samp(fit_F(x1[A], x2[A]), x1[B], x2[B])) + np.median(samp(fit_F(x1[B], x2[B]), x1[A], x2[A])))
        bg = np.median(samp(Fbg, x1[idx], x2[idx]))
        out.append(float(np.log10(max(bg, 1e-6) / max(own, 1e-6))))
    return out

# =============================================================== E5-A
res = {"A": {}, "F": {}, "P": {}, "layers": LAYERS, "parts": sorted(PARTS)}
if "A" not in PARTS:
    names_A = []
else:
    names_A = names
ATT = {n: dict(np.load(os.path.join(idir, f"{n}_att.npz"))) for n in names_A}
hits = {k: np.full((len(names), 12, 12), np.nan) for k in ["static", "track", "rigid"]}
for si, n in enumerate(names_A):
    g = GT[n]; am = ATT[n]["argmax"].astype(int)  # (12,12,T)
    for gi in range(12):
        for h in range(12):
            a = am[gi, h]
            if g["S"].sum():
                hits["static"][si, gi, h] = cheb_hit(a[g["S"]], g["x2_true"][g["S"]]).mean()
            if g["M"].sum() >= 5:
                hits["track"][si, gi, h] = cheb_hit(a[g["M"]], g["x2_true"][g["M"]]).mean()
                hits["rigid"][si, gi, h] = cheb_hit(a[g["M"]], g["x2_static"][g["M"]]).mean()
splits = np.array([GT[n]["split"] for n in names])
res["A"]["layers"] = [2 * gi for gi in range(12)]
for k, v in hits.items():
    res["A"][f"{k}_mean_heads"] = np.nanmean(np.nanmean(v, 2), 0).tolist()
    res["A"][f"{k}_max_heads"] = np.nanmax(np.nanmean(v, 0), 1).tolist()
# head selected per layer by static hit on the other fold; then track/rigid of that head on this fold
sel_track, sel_rigid, gains_att, gains_out, gains_gt = (np.full((len(names), 12), np.nan) for _ in range(5))
for si, n in enumerate(names_A):
    other = [j for j, m in enumerate(names) if fold2[m] != fold2[n]]
    g = GT[n]
    for gi in range(12):
        h = int(np.nanargmax(np.nanmean(hits["static"][other, gi], 0)))
        sel_track[si, gi] = hits["track"][si, gi, h]; sel_rigid[si, gi] = hits["rigid"][si, gi, h]
        x2a = ATT[n]["soft"][gi, h].astype(float) * PATCH + 6.5
        keep = g["S"] | (g["pure"] & (g["body"] > 0))
        gl = multibody_gain(g["c518"][keep], x2a[keep], g["body"][keep], g["S"][keep])
        gains_att[si, gi] = np.mean(gl) if gl else np.nan
    keep = g["S"] | (g["pure"] & (g["body"] > 0))
    go = multibody_gain(g["c518"][keep], g["x2_pred"][keep], g["body"][keep], g["S"][keep])
    gg = multibody_gain(g["c518"][keep], g["x2_true"][keep], g["body"][keep], g["S"][keep])
    gains_out[si] = np.mean(go) if go else np.nan; gains_gt[si] = np.mean(gg) if gg else np.nan
nonepi = splits != "epi_aligned"
res["A"]["sel_track"] = np.nanmean(sel_track, 0).tolist(); res["A"]["sel_rigid"] = np.nanmean(sel_rigid, 0).tolist()
res["A"]["sel_track_by_scene"] = sel_track.tolist(); res["A"]["sel_rigid_by_scene"] = sel_rigid.tolist()
res["A"]["log10_gain_att_nonepi_median"] = np.nanmedian(gains_att[nonepi], 0).tolist()
res["A"]["log10_gain_att_epi_median"] = np.nanmedian(gains_att[~nonepi], 0).tolist()
res["A"]["log10_gain_out_nonepi_median"] = float(np.nanmedian(gains_out[nonepi, 0]))
res["A"]["log10_gain_gt_nonepi_median"] = float(np.nanmedian(gains_gt[nonepi, 0]))
res["A"]["log10_gain_out_epi_median"] = float(np.nanmedian(gains_out[~nonepi, 0]))
res["A"]["log10_gain_gt_epi_median"] = float(np.nanmedian(gains_gt[~nonepi, 0]))
res["A"]["gain_att_by_scene"] = gains_att.tolist(); res["A"]["gain_out_by_scene"] = gains_out[:, 0].tolist()
res["A"]["gain_gt_by_scene"] = gains_gt[:, 0].tolist()
res["A"]["n_moving_distinguishable"] = {n: int(GT[n]["M"].sum()) for n in names}
print("E5-A done", flush=True)

# =============================================================== load tokens for pure patches
TOK = {}
_sl = os.path.join(idir, "stored_layers.json")
STORED = json.load(open(_sl)) if os.path.exists(_sl) else list(range(25))
assert set(LAYERS) <= set(STORED), f"requested layers {LAYERS} not all stored {STORED}"


class _LayerView:
    """Index a stored-layer subset by true residual layer number."""
    def __init__(self, arr):
        self.arr = arr; self.pos = {l: i for i, l in enumerate(LAYERS)}
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.arr[(self.pos[key[0]],) + key[1:]]
        return self.arr[self.pos[key]]


for n in names:
    a = np.load(os.path.join(idir, f"{n}_tok.npy"), mmap_mode="r")
    keep = [STORED.index(l) for l in LAYERS]
    sub = np.asarray(a[keep][:, :, GT[n]["pure"]]).astype(np.float32)
    TOK[n] = _LayerView(sub)  # (len(LAYERS), 2, P, 768), indexed by true layer number
    GT[n]["pidx"] = np.where(GT[n]["pure"])[0]

# =============================================================== E5-F
def balanced_idx(inst, cap_obj=120, cap_bg=240):
    out = []
    for i in sorted(set(inst)):
        ii = np.where(inst == i)[0]
        cap = cap_bg if i < 0 else cap_obj
        out.append(rng.choice(ii, min(cap, len(ii)), replace=False))
    return np.concatenate(out)

def pca_feats(X, d=32):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :d] * s[:d]
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

F_rows = []
for n, subset in [(n, s_) for n in (names if "F" in PARTS else []) for s_ in ("all", "objects")]:
    g = GT[n]; P = g["pidx"]; body = g["body"][P]; inst = g["inst"][P]
    sel = balanced_idx(inst)
    if subset == "objects":
        sel = sel[inst[sel] >= 0]
    b, ins = body[sel], inst[sel]
    k = len(set(b))
    if k < 2:
        continue
    for l in LAYERS:
        for vname, X in [("v0", TOK[n][l, 0, sel]), ("v6", TOK[n][l, 1, sel]),
                         ("cat", np.concatenate([TOK[n][l, 0, sel], TOK[n][l, 1, sel]], 1))]:
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(pca_feats(X))
            row = dict(scene=n, split=g["split"], subset=subset, layer=l, view=vname, ari_body=adjusted_rand_score(b, lab),
                       ari_inst=adjusted_rand_score(ins, lab))
            if g["split"] == "confound_same":
                movers = sorted(set(ins[b > 0])); statics = [i for i in set(ins[b == 0]) if i >= 0]
                # the static copy is the static instance with the same asset; take the static object instance with most patches
                if movers and statics:
                    A = lab[ins == movers[0]]; Bc = [lab[ins == s_] for s_ in statics]
                    # pick static instance whose cluster pattern differs most is unbiased? use all static objects -> report min over them
                    row["split_copies"] = float(np.mean([np.mean(A[:, None] != Bi[None, :]) for Bi in Bc]))
            if g["split"] == "confound_glued":
                movers = sorted(set(ins[b > 0]))
                if len(movers) >= 2:
                    A = lab[ins == movers[0]]; Bi = lab[ins == movers[1]]
                    row["join_glued"] = float(np.mean(A[:, None] == Bi[None, :]))
            F_rows.append(row)
res["F"]["rows"] = F_rows
print("E5-F done", flush=True)

# =============================================================== E5-P
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folds = {n: i % 5 for i, n in enumerate(sorted(names, key=lambda m: (GT[m]["split"], seed_of[m])))}

def feats(n, l):
    return np.concatenate([TOK[n][l, 0], TOK[n][l, 1]], 1)

def pairs_for(n, m, rng):
    b = GT[n]["body"][GT[n]["pidx"]]
    P = len(b)
    i = rng.integers(0, P, 4 * m); j = rng.integers(0, P, 4 * m)
    y = (b[i] == b[j])
    pos = np.where(y)[0][:m // 2]; neg = np.where(~y)[0][:m // 2]
    k = np.r_[pos, neg]
    return i[k], j[k], y[k].astype(np.float32)

def train_pair_probe(Xs, pairs, d=64, steps=None):
    steps = steps or STEPS
    D = Xs[0].shape[1]
    W = torch.nn.Linear(D, d, bias=False).to(dev); ab = torch.nn.Parameter(torch.tensor([1.0, 0.0], device=dev))
    opt = torch.optim.Adam(list(W.parameters()) + [ab], lr=1e-3, weight_decay=1e-4)
    offs = np.cumsum([0] + [len(x) for x in Xs])[:-1]
    X = torch.tensor(np.concatenate(Xs), device=dev)
    I = torch.tensor(np.concatenate([i + o for (i, j, y), o in zip(pairs, offs)]), device=dev)
    J = torch.tensor(np.concatenate([j + o for (i, j, y), o in zip(pairs, offs)]), device=dev)
    Y = torch.tensor(np.concatenate([y for (i, j, y) in pairs]), device=dev)
    for st in range(steps):
        z = W(X); s = (z[I] * z[J]).sum(1) / np.sqrt(d) * ab[0] + ab[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return W, ab

def linkage_k(A, min_size=5):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Dm = 1 - (A + A.T) / 2; np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=0.5, criterion="distance")
    _, cnt = np.unique(lab, return_counts=True)
    return int((cnt >= min_size).sum())

def eigengap_k(A, kmax=5):
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    dd = A.sum(1) + 1e-9
    L = np.eye(len(A)) - A / np.sqrt(dd[:, None] * dd[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[: kmax + 1]
    return int(np.argmax(np.diff(ev)) + 1)

def ridge_gcv(Xtr, Ytr):
    ym = Ytr.mean(0)
    X_ = torch.tensor(Xtr, device=dev, dtype=torch.float32); Y_ = torch.tensor(Ytr - ym, device=dev, dtype=torch.float32)
    Uu, ss, Vh = torch.linalg.svd(X_, full_matrices=False); Uy = Uu.T @ Y_
    best = None
    for alpha in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        dfac = ss ** 2 / (ss ** 2 + alpha)
        resid = ((Y_ - Uu @ (dfac[:, None] * Uy)) ** 2).sum().item()
        gcv = resid / (1 - dfac.sum().item() / len(Xtr)) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, alpha)
    alpha = best[1]
    coef = (Vh.T @ ((ss / (ss ** 2 + alpha))[:, None] * Uy)).cpu().numpy()
    return np.r_[coef, ym[None, :]], alpha

P_rows = []
for l in (LAYERS if "P" in PARTS else []):
    for f in range(5):
        tr = [n for n in names if folds[n] != f]; te = [n for n in names if folds[n] == f]
        mu = np.mean(np.concatenate([feats(n, l) for n in tr]), 0); sd = np.std(np.concatenate([feats(n, l) for n in tr]), 0) + 1e-6
        Z = {n: (feats(n, l) - mu) / sd for n in tr + te}
        # --- pair probe
        prs = [pairs_for(n, 600, rng) for n in tr]
        W, ab = train_pair_probe([Z[n] for n in tr], prs)
        # --- linear probes (closed form ridge for motion and object-motion)
        Xtr = np.concatenate([Z[n] for n in tr]); ytr = np.concatenate([(GT[n]["body"][GT[n]["pidx"]] > 0).astype(float) for n in tr])
        dtr = np.concatenate([(GT[n]["x2_true"] - GT[n]["x2_static"])[GT[n]["pidx"]] for n in tr])
        R, alpha = ridge_gcv(Xtr, dtr)                       # object-motion probe on concat(v0, v6)
        Xv0 = Xtr[:, :768]
        Rm, alpha_m = ridge_gcv(Xv0, (2 * ytr - 1)[:, None])  # moving-vs-static probe on view-0 token only
        for n in te:
            g = GT[n]; P = g["pidx"]; b = g["body"][P]; ins = g["inst"][P]
            with torch.no_grad():
                z = W(torch.tensor(Z[n], device=dev)).cpu().numpy()
            S = (z @ z.T) / np.sqrt(z.shape[1]) * float(ab[0]) + float(ab[1])
            i, j, y = pairs_for(n, 600, rng)
            row = dict(scene=n, split=g["split"], layer=l, fold=f)
            row["pair_auc"] = roc_auc_score(y, S[i, j]) if 0 < y.mean() < 1 else np.nan
            if g["split"] == "confound_same":
                mv = sorted(set(ins[b > 0])); st = [s_ for s_ in set(ins[b == 0]) if s_ >= 0]
                if mv and st:
                    row["copies_separated"] = float(np.mean([np.mean(S[np.ix_(ins == mv[0], ins == s_)] < 0) for s_ in st]))
            if g["split"] == "confound_glued":
                mv = sorted(set(ins[b > 0]))
                if len(mv) >= 2:
                    row["glued_joined"] = float(np.mean(S[np.ix_(ins == mv[0], ins == mv[1])] > 0))
            sub = balanced_idx(ins)
            A = 1 / (1 + np.exp(-S[np.ix_(sub, sub)]))
            row["k_hat"] = linkage_k(A); row["k_true"] = int(len(set(b[sub])))
            obj = np.where(ins >= 0)[0]
            if len(obj) > 10:
                io = rng.choice(obj, 2000); jo = rng.choice(obj, 2000); yo = b[io] == b[jo]
                row["pair_auc_obj"] = roc_auc_score(yo, S[io, jo]) if 0 < yo.mean() < 1 else np.nan
            dh = np.c_[Z[n], np.ones(len(P))] @ R
            pm = (np.c_[Z[n][:, :768], np.ones(len(P))] @ Rm)[:, 0]
            row["motion_auc_v0"] = roc_auc_score(b > 0, pm) if 0 < (b > 0).mean() < 1 else np.nan
            ob = ins >= 0
            row["motion_auc_v0_objects"] = roc_auc_score(b[ob] > 0, pm[ob]) if ob.sum() and 0 < (b[ob] > 0).mean() < 1 else np.nan
            row["ridge_alpha"] = alpha; row["ridge_alpha_motion"] = alpha_m
            dt = (g["x2_true"] - g["x2_static"])[P]
            mv = g["M"][P]
            if mv.sum() >= 5:
                row["objmotion_slope"] = float((dh[mv] * dt[mv]).sum() / (dt[mv] ** 2).sum())
                row["objmotion_relerr"] = float(np.median(np.linalg.norm(dh[mv] - dt[mv], axis=1) / np.linalg.norm(dt[mv], axis=1)))
                if l == 0:
                    do = (g["x2_pred"] - g["x2_static"])[P]
                    row["output_slope"] = float((do[mv] * dt[mv]).sum() / (dt[mv] ** 2).sum())
                    row["output_relerr"] = float(np.median(np.linalg.norm(do[mv] - dt[mv], axis=1) / np.linalg.norm(dt[mv], axis=1)))
            P_rows.append(row)
    print("E5-P layer", l, flush=True)
res["P"]["rows"] = P_rows
json.dump(res, open(outp + "_results.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
print("done")
