"""Round 2 probe controls: helpers shared by e5_internal_analysis.py (E5_SHUFFLE) and e5_probe_mlp.py.

shuffle_labels(GT, names, mode, seed, PATCH) modifies the per-patch ground truth in place (control tasks, after
Hewitt and Liang 2019). Features are untouched; only labels move, and only within a scene:
  mode "patch"    : one random permutation per scene of all label fields over the pure patches
                    (body, inst, x2_true, x2_static, x2_pred, M, S move together). Label counts are kept exactly.
                    Any feature to label relation is destroyed, so a held-out-scene probe should sit at chance.
  mode "instance" : the rigid-body label of each object instance is permuted across the object instances of the
                    scene (background stays static). Instance membership is kept, so a probe that only groups by
                    instance identity or appearance still scores above chance on the pair task. This is the
                    type-level control (instance = type) and is the stricter of the two for the pair probe.
                    M and S are recomputed from the shuffled body labels; x2_true and x2_static are not moved.
"""
import numpy as np

FIELDS = ("body", "inst", "x2_true", "x2_static", "x2_pred", "M", "S")


def shuffle_labels(GT, names, mode, seed, PATCH):
    rng = np.random.default_rng(seed)
    for n in names:
        g = GT[n]
        P = np.where(g["pure"])[0]
        if mode == "patch":
            perm = rng.permutation(len(P))
            for k in FIELDS:
                if k in g:
                    a = np.array(g[k], copy=True); a[P] = a[P][perm]; g[k] = a
        elif mode == "instance":
            body = np.array(g["body"], copy=True); inst = g["inst"]
            objs = sorted(int(o) for o in set(inst[P].tolist()) if o >= 0)
            labs = []
            for o in objs:
                v = body[P][inst[P] == o]
                vals, cnt = np.unique(v, return_counts=True)
                labs.append(int(vals[np.argmax(cnt)]))
            perm = rng.permutation(len(objs))
            for o, j in zip(objs, perm):
                body[inst == o] = labs[j]
            g["body"] = body
            dist = np.linalg.norm(g["x2_true"] - g["x2_static"], axis=1)
            g["M"] = g["pure"] & (g["body"] > 0) & (dist > 2 * PATCH)
            g["S"] = g["pure"] & (g["body"] == 0)
        else:
            raise ValueError(mode)
    return GT
