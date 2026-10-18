"""Adjacency control for E5-F / E5-P grouping claims.
For every pair of distinct object instances (a, b) in a held-out scene, compute
  probe_join : fraction of patch pairs (a_i, b_j) with pair-probe logit > 0
  km_join    : fraction of patch pairs in the same k-means cluster (objects-only, concat features)
and record same_body and the image distance between the instances' view-0 centroids (patches).
Usage: python e5_adjacency_control.py internals gt_patches out.json
"""
import os, sys, json
os.environ.setdefault("E5_PARTS", "")  # import-only of helpers is not possible; re-implement minimal parts
import numpy as np, torch, glob
from sklearn.cluster import KMeans

idir, gdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
LAYERS = [0, 8, 16, 24]; STEPS = 300; G = 37
rng = np.random.default_rng(1)
dev = torch.device("cuda")
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(idir, "*_tok.npy")))
GT = {}
for n in names:
    g = dict(np.load(os.path.join(gdir, f"{n}_gt.npz"), allow_pickle=True)); g["split"] = str(g["split"])
    g["pure"] = (g["purity"] >= 0.8) & g["vis"]; g["pidx"] = np.where(g["pure"])[0]; GT[n] = g
seed_of = {n: int(n.split("_s")[-1]) for n in names}
folds = {n: i % 5 for i, n in enumerate(sorted(names, key=lambda m: (GT[m]["split"], seed_of[m])))}
_sl = os.path.join(idir, "stored_layers.json")
STORED = json.load(open(_sl)) if os.path.exists(_sl) else list(range(25))
LAYERS = [l for l in LAYERS if l in STORED] or [STORED[0], STORED[len(STORED) // 3], STORED[2 * len(STORED) // 3], STORED[-1]]
_pos = [STORED.index(l) for l in LAYERS]
TOK = {n: np.asarray(np.load(os.path.join(idir, f"{n}_tok.npy"), mmap_mode="r")[_pos][:, :, GT[n]["pidx"]]).astype(np.float32) for n in names}

def pairs_for(n, m):
    b = GT[n]["body"][GT[n]["pidx"]]; P = len(b)
    i = rng.integers(0, P, 4 * m); j = rng.integers(0, P, 4 * m); y = b[i] == b[j]
    k = np.r_[np.where(y)[0][: m // 2], np.where(~y)[0][: m // 2]]
    return i[k], j[k], y[k].astype(np.float32)

def train(Xs, prs, d=64):
    W = torch.nn.Linear(Xs[0].shape[1], d, bias=False).to(dev); ab = torch.nn.Parameter(torch.tensor([1.0, 0.0], device=dev))
    opt = torch.optim.Adam(list(W.parameters()) + [ab], lr=1e-3, weight_decay=1e-4)
    offs = np.cumsum([0] + [len(x) for x in Xs])[:-1]
    X = torch.tensor(np.concatenate(Xs), device=dev)
    I = torch.tensor(np.concatenate([p[0] + o for p, o in zip(prs, offs)]), device=dev)
    J = torch.tensor(np.concatenate([p[1] + o for p, o in zip(prs, offs)]), device=dev)
    Y = torch.tensor(np.concatenate([p[2] for p in prs]), device=dev)
    for _ in range(STEPS):
        z = W(X); s = (z[I] * z[J]).sum(1) / np.sqrt(d) * ab[0] + ab[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, Y); opt.zero_grad(); loss.backward(); opt.step()
    return W, ab

def pca(X, d=32):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); U, s, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :d] * s[:d]; return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

rows = []
for li, l in enumerate(LAYERS):
    for f in range(5):
        tr = [n for n in names if folds[n] != f]; te = [n for n in names if folds[n] == f]
        feat = lambda n: np.concatenate([TOK[n][li, 0], TOK[n][li, 1]], 1)
        allf = np.concatenate([feat(n) for n in tr]); mu, sd = allf.mean(0), allf.std(0) + 1e-6
        W, ab = train([(feat(n) - mu) / sd for n in tr], [pairs_for(n, 600) for n in tr])
        for n in te:
            g = GT[n]; P = g["pidx"]; b = g["body"][P]; ins = g["inst"][P]
            Z = (feat(n) - mu) / sd
            with torch.no_grad():
                z = W(torch.tensor(Z, device=dev)).cpu().numpy()
            S = z @ z.T / np.sqrt(z.shape[1]) * float(ab[0]) + float(ab[1])
            obj = np.where(ins >= 0)[0]
            insts = sorted(set(ins[obj]))
            if len(insts) < 2:
                continue
            kb = len(set(b[obj]))
            km = KMeans(max(kb, 2), n_init=10, random_state=0).fit_predict(pca(np.concatenate([TOK[n][li, 0][obj], TOK[n][li, 1][obj]], 1)))
            lab = -np.ones(len(P), int); lab[obj] = km
            rc = (P // G).astype(float); cc = (P % G).astype(float)
            for ai in range(len(insts)):
                for bi in range(ai + 1, len(insts)):
                    A = ins == insts[ai]; B = ins == insts[bi]
                    if A.sum() < 3 or B.sum() < 3:
                        continue
                    dA = np.array([rc[A].mean(), cc[A].mean()]); dB = np.array([rc[B].mean(), cc[B].mean()])
                    # minimum patch-to-patch distance (touching = 1)
                    dmin = float(np.min(np.hypot(rc[A][:, None] - rc[B][None], cc[A][:, None] - cc[B][None])))
                    rows.append(dict(scene=n, split=g["split"], layer=l, same_body=bool(b[A][0] == b[B][0]),
                                     moving_pair=int(b[A][0] > 0) + int(b[B][0] > 0), inst_a=int(insts[ai]), inst_b=int(insts[bi]), dmin=dmin,
                                     dcent=float(np.linalg.norm(dA - dB)),
                                     probe_join=float(np.mean(S[np.ix_(A, B)] > 0)),
                                     km_join=float(np.mean(lab[A][:, None] == lab[B][None]))))
        print("layer", l, "fold", f, flush=True)
json.dump(rows, open(outp, "w"))
print("done", len(rows))
