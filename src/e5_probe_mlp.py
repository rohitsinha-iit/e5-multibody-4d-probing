"""Round 2, review W4: nonlinear (MLP) probes and matched linear probes, with label-shuffle controls and seeds.

Usage: python e5_probe_mlp.py <internals_dir> <gt_dir> <out.json> --layers 0,24 --seed 0 [--shuffle patch|instance]
                              [--arch mlp,lin]

Same patch set, labels, scene folds (5-fold, grouped by scene, folds assigned as in e5_internal_analysis.py),
feature standardisation and metric definitions as part P of e5_internal_analysis.py:
  motion probe : view-0 token -> moving (body > 0) vs static; AUC per held-out scene over all pure patches
                 (motion_auc_v0) and over object patches only (motion_auc_v0_objects).
  pair probe   : concat(view-0, view-6) token; score s_ij = a * <g(x_i), g(x_j)> / sqrt(64) + b, trained with BCE on
                 same-rigid-body pairs (pairs_for, 600 per scene, half positive); AUC on held-out pairs (pair_auc)
                 and on 2000 random object-object pairs (pair_auc_obj).
Architectures (one training recipe for both, so linear vs MLP differs only in the nonlinearity):
  mlp : 2 layers, 256 hidden, ReLU.  motion: D -> 256 -> 1.  pair: g = D -> 256 -> 64.
  lin : motion: D -> 1 (logistic regression).  pair: g = D -> 64 (the low-rank bilinear probe of round 1).
Training: AdamW (lr 1e-3, weight decay 1e-2), minibatch 4096, up to 4000 steps; early stopping on the loss of a
validation split of the training scenes (15% of training scenes, grouped by scene, seeded), checked every 20 steps,
patience 10 checks; the best checkpoint is restored.
Resumable: rows are checkpointed per layer to <out.json>.partial.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_probe_common as pc

ap = argparse.ArgumentParser()
ap.add_argument("idir"); ap.add_argument("gdir"); ap.add_argument("out")
ap.add_argument("--layers", default="0,24"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shuffle", default=""); ap.add_argument("--arch", default="mlp,lin")
ap.add_argument("--max_steps", type=int, default=4000)
args = ap.parse_args()
SEED = args.seed
rng = np.random.default_rng(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = [int(x) for x in args.layers.split(",")]
ARCHS = args.arch.split(",")
_g0 = sorted(glob.glob(os.path.join(args.gdir, "*_gt.npz")))
G = int(round(np.sqrt(len(np.load(_g0[0])["body"])))); PATCH = 14 if G == 37 else 518 / G
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(args.idir, "*_tok.npy")))
GT = {n: dict(np.load(os.path.join(args.gdir, f"{n}_gt.npz"), allow_pickle=True)) for n in names}
for n in names:
    g = GT[n]; g["split"] = str(g["split"])
    g["pure"] = (g["purity"] >= 0.8) & g["vis"]
    dist = np.linalg.norm(g["x2_true"] - g["x2_static"], axis=1)
    g["M"] = g["pure"] & (g["body"] > 0) & (dist > 2 * PATCH)
    g["S"] = g["pure"] & (g["body"] == 0)
if args.shuffle:
    pc.shuffle_labels(GT, names, args.shuffle, 1000 + SEED, PATCH)
seed_of = {n: int(n.split("_s")[-1]) for n in names}
folds = {n: i % 5 for i, n in enumerate(sorted(names, key=lambda m: (GT[m]["split"], seed_of[m])))}
_sl = os.path.join(args.idir, "stored_layers.json")
STORED = json.load(open(_sl)) if os.path.exists(_sl) else list(range(25))
assert set(LAYERS) <= set(STORED), (LAYERS, STORED)
for n in names:
    GT[n]["pidx"] = np.where(GT[n]["pure"])[0]
print("scenes", len(names), "layers", LAYERS, "seed", SEED, "shuffle", args.shuffle or "none", flush=True)


def load_layer(l):
    out = {}
    for n in names:
        a = np.load(os.path.join(args.idir, f"{n}_tok.npy"), mmap_mode="r")
        out[n] = np.asarray(a[STORED.index(l)][:, GT[n]["pidx"]]).astype(np.float32)  # (2, P, D)
    return out


def pairs_for(n, m, rng):
    b = GT[n]["body"][GT[n]["pidx"]]
    P = len(b)
    i = rng.integers(0, P, 4 * m); j = rng.integers(0, P, 4 * m)
    y = (b[i] == b[j])
    pos = np.where(y)[0][:m // 2]; neg = np.where(~y)[0][:m // 2]
    k = np.r_[pos, neg]
    return i[k], j[k], y[k].astype(np.float32)


def make_net(D, out, arch):
    if arch == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, out)).to(dev)
    return torch.nn.Linear(D, out).to(dev)


def fit(params, loss_fn, n_tr, n_va, val_loss_fn, bs=4096):
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    best, best_state, bad, step = float("inf"), None, 0, 0
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(0, n_tr, (min(bs, n_tr),), device=dev)
        loss = loss_fn(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                vl = float(val_loss_fn())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= 10:
                    break
    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.copy_(s)
    return step, best


bce = torch.nn.functional.binary_cross_entropy_with_logits
rows = json.load(open(args.out + ".partial"))["rows"] if os.path.exists(args.out + ".partial") else []
done_layers = {r["layer"] for r in rows}
for l in LAYERS:
    if l in done_layers:
        continue
    t0 = time.time()
    TOK = load_layer(l)
    DV = TOK[names[0]].shape[-1]
    for f in range(5):
        tr_all = [n for n in names if folds[n] != f]; te = [n for n in names if folds[n] == f]
        vrng = np.random.default_rng(SEED * 100 + f)
        va = list(vrng.choice(tr_all, max(2, int(round(0.15 * len(tr_all)))), replace=False)); tr = [n for n in tr_all if n not in va]
        cat = {n: np.concatenate([TOK[n][0], TOK[n][1]], 1) for n in names}
        Xall = np.concatenate([cat[n] for n in tr_all]); mu = Xall.mean(0); sd = Xall.std(0) + 1e-6; del Xall
        Z = {n: torch.tensor((cat[n] - mu) / sd, device=dev) for n in names}
        del cat
        ylab = {n: torch.tensor((GT[n]["body"][GT[n]["pidx"]] > 0).astype(np.float32), device=dev) for n in names}
        # ---------------- motion probe (view-0 token)
        Xtr = torch.cat([Z[n][:, :DV] for n in tr]); ytr = torch.cat([ylab[n] for n in tr])
        Xva = torch.cat([Z[n][:, :DV] for n in va]); yva = torch.cat([ylab[n] for n in va])
        prs_tr = [pairs_for(n, 600, rng) for n in tr]; prs_va = [pairs_for(n, 600, rng) for n in va]

        def stack_pairs(ns, prs):
            X = torch.cat([Z[n] for n in ns]); offs = np.cumsum([0] + [len(Z[n]) for n in ns])[:-1]
            I = torch.tensor(np.concatenate([i + o for (i, j, y), o in zip(prs, offs)]), device=dev)
            J = torch.tensor(np.concatenate([j + o for (i, j, y), o in zip(prs, offs)]), device=dev)
            Y = torch.tensor(np.concatenate([y for (i, j, y) in prs]), device=dev)
            return X, I, J, Y
        PXtr, PItr, PJtr, PYtr = stack_pairs(tr, prs_tr); PXva, PIva, PJva, PYva = stack_pairs(va, prs_va)
        test_pairs = {n: pairs_for(n, 600, rng) for n in te}
        test_obj = {}
        for n in te:
            ins = GT[n]["inst"][GT[n]["pidx"]]; obj = np.where(ins >= 0)[0]
            test_obj[n] = (rng.choice(obj, 2000), rng.choice(obj, 2000)) if len(obj) > 10 else None
        for arch in ARCHS:
            # motion
            net = make_net(DV, 1, arch)
            st_m, _ = fit(list(net.parameters()), lambda idx: bce(net(Xtr[idx])[:, 0], ytr[idx]), len(Xtr), len(Xva),
                          lambda: bce(net(Xva)[:, 0], yva))
            # pair
            emb = make_net(2 * DV, 64, arch); ab = torch.nn.Parameter(torch.tensor([1.0, 0.0], device=dev))

            def pscore(X, I, J):
                zi = emb(X[I]); zj = emb(X[J])
                return (zi * zj).sum(1) / 8.0 * ab[0] + ab[1]
            st_p, _ = fit(list(emb.parameters()) + [ab], lambda idx: bce(pscore(PXtr, PItr[idx], PJtr[idx]), PYtr[idx]),
                          len(PItr), len(PIva), lambda: bce(pscore(PXva, PIva, PJva), PYva))
            with torch.no_grad():
                for n in te:
                    g = GT[n]; P = g["pidx"]; b = g["body"][P]; ins = g["inst"][P]
                    row = dict(scene=n, split=g["split"], layer=l, fold=f, seed=SEED, shuffle=args.shuffle or "none",
                               arch=arch, steps_motion=st_m, steps_pair=st_p)
                    pm = net(Z[n][:, :DV])[:, 0].cpu().numpy()
                    row["motion_auc_v0"] = roc_auc_score(b > 0, pm) if 0 < (b > 0).mean() < 1 else float("nan")
                    ob = ins >= 0
                    row["motion_auc_v0_objects"] = roc_auc_score(b[ob] > 0, pm[ob]) if ob.sum() and 0 < (b[ob] > 0).mean() < 1 else float("nan")
                    zz = emb(Z[n]); S = ((zz @ zz.T) / 8.0 * ab[0] + ab[1]).cpu().numpy()
                    i, j, y = test_pairs[n]
                    row["pair_auc"] = roc_auc_score(y, S[i, j]) if 0 < y.mean() < 1 else float("nan")
                    if test_obj[n] is not None:
                        io, jo = test_obj[n]; yo = b[io] == b[jo]
                        row["pair_auc_obj"] = roc_auc_score(yo, S[io, jo]) if 0 < yo.mean() < 1 else float("nan")
                    rows.append(row)
        del Z, PXtr, PXva, Xtr, Xva
        torch.cuda.empty_cache()
    json.dump(dict(rows=rows), open(args.out + ".partial.tmp", "w")); os.replace(args.out + ".partial.tmp", args.out + ".partial")
    print("layer", l, f"{time.time()-t0:.0f}s", flush=True)
json.dump(dict(rows=rows, layers=LAYERS, seed=SEED, shuffle=args.shuffle or None, archs=ARCHS), open(args.out + ".tmp", "w"))
os.replace(args.out + ".tmp", args.out)
print("done", len(rows))
