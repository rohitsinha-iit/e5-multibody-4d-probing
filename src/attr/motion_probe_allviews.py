#!/usr/bin/env python3
"""All-view internal motion probe for Any4D (unified plan 3.7; used by U5b two-pass masking).

Label: token lies on a moving object at its own view. D tokens: GT moving_mask of D at the token grid (>= 50% of the
token's pixels). B tokens (objects frozen, same objects present): all labelled 0, so B supplies hard negatives.
Features: residual stream after block L, L* in {13,15,17,19,21} (block outputs, 0-indexed; odd = frame blocks).
Steps:
  extract  fit scenes -> $E5_BIG/attr/probe_feats/any4d/<scene>.npz (all positives + up to --n_neg negatives per variant,
           fp16, per layer), plus the token labels and positions.
  fit      5-fold CV grouped by scene for every L (logistic regression, standardised features, L2 1e-3, full-batch
           LBFGS in torch); L* = best mean held-out AUC; refit on all fit scenes; threshold = 90% recall on the
           out-of-fold scores at L*. Writes $E5_BIG/results/u5b/probe_any4d.json and probe_any4d.pt.
  eval     on given scenes (split rule as interchange.py): AUC over all tokens of D and B, and mask IoU of the predicted
           token mask vs the GT token mask (D, all 7 views; B false-positive rate).
Library use: probe_mask(H, views, probe) runs pass 1 (capture block L* only) and returns the bool (V*P,) key mask.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import any4d_hooks as ah  # noqa
import interchange as ic  # noqa

E5_BIG = ic.E5_BIG
LAYERS = [13, 15, 17, 19, 21]
FEAT = os.environ.get("PROBE_FEAT", os.path.join(E5_BIG, "attr", "probe_feats", "any4d"))
OUT = os.environ.get("U5B_RES", os.path.join(E5_BIG, "results", "u5b"))


def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # ties: average ranks
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def fit_logreg(X, y, l2=1e-3, iters=200):
    X = torch.as_tensor(X, dtype=torch.float32, device="cuda"); y = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4); Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
    pw = (1 - y).sum() / y.sum().clamp_min(1)    # class balance
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lg = Z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, y, pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return {"mu": mu.detach().cpu(), "sd": sd.detach().cpu(), "w": w.detach().cpu(), "b": b.detach().cpu()}


def score(pr, X):
    X = torch.as_tensor(X, dtype=torch.float32)
    return (((X - pr["mu"]) / pr["sd"]) @ pr["w"] + pr["b"]).numpy()


def extract(H, scenes, n_neg=3000, seed=0):
    os.makedirs(FEAT, exist_ok=True); rng = np.random.default_rng(seed)
    for d in scenes:
        f = os.path.join(FEAT, ic.sid(d) + ".npz")
        if os.path.exists(f):
            continue
        ro_lab = {v: ah.token_mask_from_pixels(ic.gt_of(d, v)["moving_mask"][ah.FRAMES], 0.5).reshape(-1).numpy() for v in ("D",)}
        rec = {}
        for v in ("D", "B"):
            _, cache = H.run(ah.load_views(os.path.join(d, v)), capture=True, capture_only=[("r", L) for L in LAYERS])
            lab = ro_lab["D"] if v == "D" else np.zeros_like(ro_lab["D"])
            pos = np.nonzero(lab)[0]; neg = np.nonzero(~lab)[0]
            neg = rng.choice(neg, min(n_neg, len(neg)), replace=False)
            idx = np.sort(np.concatenate([pos, neg]))
            for L in LAYERS:
                rec[f"{v}_L{L}"] = cache[("r", L)][idx].half().cpu().numpy()
            rec[f"{v}_idx"] = idx; rec[f"{v}_y"] = lab[idx]
        np.savez(f + ".tmp.npz", **rec); os.replace(f + ".tmp.npz", f)
        print("[probe] extracted", ic.sid(d), flush=True)


def load_feats(scenes, L):
    X, y, g = [], [], []
    for i, d in enumerate(scenes):
        z = np.load(os.path.join(FEAT, ic.sid(d) + ".npz"))
        for v in ("D", "B"):
            X.append(z[f"{v}_L{L}"].astype(np.float32)); y.append(z[f"{v}_y"]); g.append(np.full(len(z[f"{v}_y"]), i))
    return np.concatenate(X), np.concatenate(y).astype(bool), np.concatenate(g)


def fit(scenes, folds=5):
    res = {"layers": {}, "n_scenes": len(scenes)}
    best, oof_best = None, None
    for L in LAYERS:
        X, y, g = load_feats(scenes, L)
        nf = min(folds, len(scenes)); fold = g % nf; oof = np.zeros(len(y))
        for k in range(nf):
            tr = fold != k
            if y[tr].sum() == 0 or (fold == k).sum() == 0:
                continue
            pr = fit_logreg(X[tr], y[tr]); oof[fold == k] = score(pr, X[fold == k])
        a = auc(y, oof); res["layers"][L] = {"cv_auc": a, "n": int(len(y)), "n_pos": int(y.sum())}
        print(f"[probe] L{L} cv AUC {a:.4f}", flush=True)
        if best is None or a > res["layers"][best]["cv_auc"]:
            best, oof_best, y_best = L, oof, y
    thr = float(np.quantile(oof_best[y_best], 0.10))   # 90% recall on out-of-fold fit scores
    X, y, _ = load_feats(scenes, best); pr = fit_logreg(X, y)
    res.update(L_star=best, threshold=thr, recall_at_thr_oof=float((oof_best[y_best] >= thr).mean()),
               fpr_at_thr_oof=float((oof_best[~y_best] >= thr).mean()), scenes=[ic.sid(d) for d in scenes])
    os.makedirs(OUT, exist_ok=True)
    torch.save(dict(pr, L=best, threshold=thr), os.path.join(OUT, "probe_any4d.pt"))
    json.dump(res, open(os.path.join(OUT, "probe_any4d.json"), "w"), indent=1)
    return res


def load_probe(path=None):
    return torch.load(path or os.path.join(OUT, "probe_any4d.pt"))


def probe_scores(H, views, probe):
    _, cache = H.run(views, capture=True, capture_only=[("r", probe["L"])])
    X = cache[("r", probe["L"])].float()
    s = ((X - probe["mu"].cuda()) / probe["sd"].cuda()) @ probe["w"].cuda() + probe["b"].cuda()
    return s


def probe_mask(H, views, probe):
    """Pass 1 of the two-pass inference: bool (V*P,) mask of tokens the probe flags as moving."""
    return probe_scores(H, views, probe) >= probe["threshold"]


def evaluate(H, scenes, probe):
    rows = []
    for d in scenes:
        lab = ah.token_mask_from_pixels(ic.gt_of(d, "D")["moving_mask"][ah.FRAMES], 0.5).reshape(-1).numpy()
        sD = probe_scores(H, ah.load_views(os.path.join(d, "D")), probe).cpu().numpy()
        sB = probe_scores(H, ah.load_views(os.path.join(d, "B")), probe).cpu().numpy()
        pm = sD >= probe["threshold"]
        inter = (pm & lab).sum(); uni = (pm | lab).sum()
        rows.append(dict(scene=ic.sid(d), auc_D=auc(lab, sD) if lab.any() else float("nan"),
                         auc_DB=auc(np.r_[lab, np.zeros_like(lab)], np.r_[sD, sB]),
                         iou_D=float(inter / max(uni, 1)), recall_D=float((pm & lab).sum() / max(lab.sum(), 1)),
                         fpr_B=float((sB >= probe["threshold"]).mean()), frac_flag_D=float(pm.mean())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["extract", "fit", "eval", "all"])
    p.add_argument("--split", default="fit"); p.add_argument("--allow_test", action="store_true")
    p.add_argument("--scenes", default=""); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_tag", default="")
    a = p.parse_args()
    if a.split in ("test", "transfer") and not a.allow_test:
        sys.exit("test/transfer split requires --allow_test")
    if a.mode in ("extract", "fit", "all") and a.split not in ("fit", "any"):
        sys.exit("the probe is fitted on fit scenes only")
    scenes = ic.split_scenes(a.split)
    if a.scenes:
        w = set(a.scenes.split(",")); scenes = [d for d in scenes if ic.sid(d) in w]
    if a.limit:
        scenes = scenes[:a.limit]
    print(f"[probe] mode={a.mode} split={a.split} scenes={len(scenes)}", flush=True)
    H = ah.Any4DHooks(ah.load_any4d())
    if a.mode in ("extract", "all"):
        extract(H, scenes)
    if a.mode in ("fit", "all"):
        r = fit(scenes); print(json.dumps({k: r[k] for k in ("L_star", "threshold", "recall_at_thr_oof", "fpr_at_thr_oof")}))
    if a.mode == "eval":
        rows = evaluate(H, scenes, load_probe())
        os.makedirs(OUT, exist_ok=True)
        f = os.path.join(OUT, f"probe_eval_any4d_{a.split}{a.out_tag}.json"); json.dump(rows, open(f, "w"), indent=1)
        print(f, {k: float(np.nanmedian([r[k] for r in rows])) for k in ("auc_D", "auc_DB", "iou_D", "fpr_B")})


if __name__ == "__main__":
    main()
