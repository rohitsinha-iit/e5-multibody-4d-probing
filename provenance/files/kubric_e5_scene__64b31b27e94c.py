"""E5 controlled multibody scenes for Kubric (kubricdockerhub/kubruntu, kubric 2022.4.1, Blender 2.93).

Rigid, scripted (kinematic) motion only: no physics. GT motion groups:
  group 0  = background dome + all static objects (moves rigidly with the world)
  group k  = k-th independently moving rigid body (glued objects share a group)
Splits
  n2_generic      arc camera, 1 moving object (rotation + translation), 3 static
  n3_generic      arc camera, 2 moving objects, 2 static
  camrot          camera pure rotation (pan about its centre), 1 moving object, 3 static
  epi_aligned     camera translates sideways with fixed orientation; 1 object translates
                  parallel to the camera translation (motion along epipolar lines; shared epipole)
  planar          arc camera, 1 moving thin textured card, 2 static
  confound_same   arc camera, two copies of the same asset: one moving, one static; + 2 static
  confound_glued  arc camera, two different objects glued into one rigid body; + 2 static
  confound_farglued (v3, C3 adjacency control) two different objects A, B rigidly coupled but far apart
                  (centre distance --far_lo..--far_hi m, invisible rigid offset) = group 1, plus an identical
                  copy C of asset A with its own independent motion = group 2, + 1 static.
                  Appearance says (A,C) belong together, motion says (A,B).
v3 float set: --float_height h > 0 lifts every object (moving and static) to a random height in [h, h+0.5] m
above the floor with the same planar motion as v2 (xy translation + yaw), and --no_shadows 1 disables shadow
casting for all objects, so neither floor contact nor a floor shadow reveals motion.
Outputs (job-dir): rgba_XXXXX.png, gt.npz (depth = ray distance, segmentation, forward_flow),
e5_meta.json (K in pixels, per-frame 4x4 world matrices of camera and objects, motion groups).
"""
import json
import logging
import numpy as np
import bpy
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

SPLITS = ["n2_generic", "n3_generic", "camrot", "epi_aligned", "planar",
          "confound_same", "confound_glued", "confound_farglued"]

parser = kb.ArgumentParser()
parser.add_argument("--split", type=str, required=True, choices=SPLITS)
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--grounded", type=int, default=0,
                    help="v2: moving objects rest on the floor and slide (planar rigid motion: xy translation + yaw)")
parser.add_argument("--float_height", type=float, default=0.0,
                    help="v3: >0 lifts all objects to [h, h+0.5] m above the floor (use with --grounded 1 motion)")
parser.add_argument("--no_shadows", type=int, default=0, help="v3: objects cast no shadows")
parser.add_argument("--far_lo", type=float, default=2.4, help="confound_farglued: min centre distance (m)")
parser.add_argument("--far_hi", type=float, default=3.2, help="confound_farglued: max centre distance (m)")
parser.add_argument("--size_lo", type=float, default=1.2)
parser.add_argument("--size_hi", type=float, default=2.0)
parser.add_argument("--min_sep_px", type=float, default=20.0,
                    help="min two-view multibody separation (px, frame first->last) for non-degenerate splits")
parser.add_argument("--kubasic_assets", type=str, default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str, default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json")
parser.set_defaults(frame_end=24, frame_rate=12, resolution="512x512")
FLAGS = parser.parse_args()

scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
W, H = scene.resolution
F0, F1 = FLAGS.frame_start, FLAGS.frame_end
FRAMES = list(range(F0, F1 + 1))
TAU = {f: (f - F0) / max(F1 - F0, 1) for f in FRAMES}
renderer = KubricRenderer(scene, scratch_dir, use_denoising=True, samples_per_pixel=FLAGS.samples)

kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
gso_train, _ = gso.get_test_split(fraction=0.1)
hdri_train, _ = hdri_source.get_test_split(fraction=0.1)

# ------------------------------------------------------------------ background
hdri_id = rng.choice(hdri_train)
background_hdri = hdri_source.create(asset_id=hdri_id)
renderer._set_ambient_light_hdri(background_hdri.filename)
dome = kubasic.create(asset_id="dome", name="dome", static=True, background=True)
scene += dome
dome_blender = dome.linked_objects[renderer]
dome_blender.data.materials[0].node_tree.nodes["Image Texture"].image = \
    bpy.data.images.load(background_hdri.filename)

# ------------------------------------------------------------------ helpers
def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def rand_unit(rng):
    v = rng.normal(size=3); return v / np.linalg.norm(v)

FOCAL, SENSOR = 35.0, 32.0
fx = FOCAL / SENSOR * W

def cam_pose_lookat(pos, target):
    f = np.asarray(target, float) - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f

def project(X, cam):  # cam = (pos, r, u, f) -> pixel (x right, y down), depth
    d = X - cam[0]
    zc = d @ cam[3]
    return np.array([fx * (d @ cam[1]) / zc + W / 2, -fx * (d @ cam[2]) / zc + H / 2]), zc

# ------------------------------------------------------------------ camera path
split = FLAGS.split
cam_target = np.array([0.0, 0.0, 0.75])
radius, height = rng.uniform(9, 11), rng.uniform(4, 5.5)
th0 = rng.uniform(0, 2 * np.pi)
cam_path = {}
if split == "camrot":
    pos = np.array([radius * np.cos(th0), radius * np.sin(th0), height])
    side = cam_pose_lookat(pos, cam_target)[0]
    pan = rng.uniform(2.5, 3.5)  # metres of target sweep at the scene centre (~15-20 deg)
    for f in FRAMES:
        cam_path[f] = (pos, cam_target + side * pan * (TAU[f] - 0.5))
elif split == "epi_aligned":
    pos0 = np.array([radius * np.cos(th0), radius * np.sin(th0), height])
    r0, _, _ = cam_pose_lookat(pos0, cam_target)
    cam_shift = rng.uniform(2.5, 3.5) * r0  # sideways, orientation fixed
    for f in FRAMES:
        cam_path[f] = (pos0 + cam_shift * (TAU[f] - 0.5), cam_target + cam_shift * 0 + 0)  # target fixed below
else:
    dth = np.deg2rad(rng.uniform(25, 40)) * rng.choice([-1, 1])
    for f in FRAMES:
        th = th0 + dth * (TAU[f] - 0.5)
        cam_path[f] = (np.array([radius * np.cos(th), radius * np.sin(th), height]), cam_target)

scene.camera = kb.PerspectiveCamera(focal_length=FOCAL, sensor_width=SENSOR)
if split == "epi_aligned":
    scene.camera.position = cam_path[F0][0]
    scene.camera.look_at(cam_target)
    q_fixed = np.array(scene.camera.quaternion)
cams = {}
for f in FRAMES:
    pos, tgt = cam_path[f]
    scene.camera.position = pos
    if split == "epi_aligned":
        scene.camera.quaternion = q_fixed
        r, u, fw = cam_pose_lookat(cam_path[F0][0], cam_target)
    else:
        scene.camera.look_at(tgt)
        r, u, fw = cam_pose_lookat(pos, tgt)
    scene.camera.keyframe_insert("position", f)
    scene.camera.keyframe_insert("quaternion", f)
    cams[f] = (pos, r, u, fw)

# ------------------------------------------------------------------ objects
def new_gso(asset_id=None, size=None):
    asset_id = asset_id or rng.choice(gso_train)
    obj = gso.create(asset_id=asset_id)
    size = size or rng.uniform(FLAGS.size_lo, FLAGS.size_hi)
    obj.scale = size / np.max(obj.bounds[1] - obj.bounds[0])
    obj.metadata["asset_id"] = asset_id
    obj.metadata["size"] = float(size)
    return obj, asset_id, size

def new_card(size=None):
    size = size or (rng.uniform(FLAGS.size_lo, FLAGS.size_hi) if FLAGS.grounded else rng.uniform(1.6, 2.2))
    card_scale = (size / 2, 0.01, size / 2 * 0.7) if FLAGS.grounded else (size / 2, size / 2 * 0.7, 0.01)  # upright when grounded
    obj = kb.Cube(name="card", scale=card_scale,
                  material=kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng)))
    obj.metadata["asset_id"] = "card"
    obj.metadata["size"] = float(size)
    return obj, "card", size

def texture_card(obj):
    """Give the card an image texture (a different HDRI) so it has matchable texture."""
    try:
        tex = hdri_source.create(asset_id=rng.choice([h for h in hdri_train if h != hdri_id]))
        bobj = obj.linked_objects[renderer]
        mat = bpy.data.materials.new("card_tex"); mat.use_nodes = True
        nodes = mat.node_tree.nodes
        img = nodes.new("ShaderNodeTexImage"); img.image = bpy.data.images.load(tex.filename)
        mat.node_tree.links.new(img.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
        bobj.data.materials.clear(); bobj.data.materials.append(mat)
        return True
    except Exception as e:  # keep the coloured card if texturing fails
        logging.warning("card texture failed: %s", e)
        return False

placed = []  # (center trajectory dict f->xyz, radius)

def clear_of_others(traj, rad):
    for t2, r2 in placed:
        for f in FRAMES:
            if np.linalg.norm(traj[f][:2] - t2[f][:2]) < 0.75 * (rad + r2):
                return False
    return True

def visible(traj, margin=0.12):
    for f in FRAMES:
        p, z = project(traj[f], cams[f])
        if z < 1 or not (margin * W < p[0] < (1 - margin) * W and margin * H < p[1] < (1 - margin) * H):
            return False
    return True

def place_static(obj, size, near=None):
    for _ in range(300):
        xy = rng.uniform(-3.0, 3.0, 2) if near is None else near + rng.uniform(-2.5, 2.5, 2)
        zb = -obj.bounds[0][2] * obj.scale[2] + 0.005 + lift()
        c = np.r_[xy, zb]
        traj = {f: c for f in FRAMES}
        if visible(traj, 0.08) and clear_of_others(traj, size / 2):
            obj.position = c
            obj.quaternion = quat_axis_angle([0, 0, 1], rng.uniform(0, 2 * np.pi))
            placed.append((traj, size / 2))
            return True
    return False

def lift():
    """v3 float set: extra height above the floor (0 for v1/v2)."""
    return float(rng.uniform(FLAGS.float_height, FLAGS.float_height + 0.5)) if FLAGS.float_height > 0 else 0.0

def rest_height(obj):
    """Height of the object origin above the floor when upright (yaw-only rotation)."""
    if obj.metadata.get("asset_id") == "card":
        return float(obj.scale[2]) + 0.005
    return float(-obj.bounds[0][2] * obj.scale[2]) + 0.005

def motion_params(kind):
    if FLAGS.grounded and kind != "epi":
        v = rand_unit(rng); v[2] = 0; v = v / np.linalg.norm(v) * rng.uniform(1.5, 3.0)
        return v, np.array([0, 0, 1.0]), np.deg2rad(rng.uniform(30, 90)) * rng.choice([-1, 1])
    if kind == "epi":
        v = -rng.uniform(0.6, 1.0) * cam_shift  # parallel to camera translation, opposite sense
        return v, np.array([0, 0, 1.0]), 0.0
    v = rand_unit(rng) * rng.uniform(1.5, 3.0); v[2] *= 0.3
    return v, rand_unit(rng), np.deg2rad(rng.uniform(30, 90)) * rng.choice([-1, 1])

def kinematics(members, kind):
    """members: list of (obj, size, offset from pivot). Returns per-frame (pos, quat) per member."""
    for _ in range(4000):
        pivot = np.r_[rng.uniform(-2.5, 2.5, 2), 0.0 if FLAGS.grounded else rng.uniform(1.2, 2.0)]
        v, axis, ang = motion_params(kind)
        q0s = [quat_axis_angle(rand_unit(rng), rng.uniform(0, np.pi)) if (kind != "epi" and not FLAGS.grounded)
               else quat_axis_angle([0, 0, 1], rng.uniform(0, 2 * np.pi)) for _ in members]
        if FLAGS.grounded:  # members rest on the floor (v3 float: lifted): offset z = rest height, yaw only
            if not hasattr(kinematics, "lifts") or len(kinematics.lifts) != len(members):
                kinematics.lifts = [lift() for _ in members]
            members = [(o_, s_, np.r_[off_[:2], rest_height(o_) + lz]) for (o_, s_, off_), lz in zip(members, kinematics.lifts)]
        out, ok = [], True
        trajs = []
        for (obj, size, off), q0 in zip(members, q0s):
            tr = {}
            for f in FRAMES:
                qR = quat_axis_angle(axis, ang * TAU[f])
                tr[f] = (quat_to_R(qR) @ off + pivot + v * TAU[f], quat_mul(qR, q0))
            trajs.append(tr)
        rad = max(s for _, s, _ in members) / 2 + max(np.linalg.norm(o) for *_, o in members)
        ctr = {f: np.mean([t[f][0] for t in trajs], 0) for f in FRAMES}
        far = len(members) > 1 and max(np.linalg.norm(o[:2]) for *_, o in members) > 1.0
        if far:  # far-apart rigid members: check each member separately, not one big circle
            ok_far = all(visible({f: t[f][0] for f in FRAMES}) and clear_of_others({f: t[f][0] for f in FRAMES}, s_ / 2)
                         for t, (_, s_, _) in zip(trajs, members))
        if far and ok_far:
            body = (v, axis, ang, pivot)
            for t, (_, s_, _) in zip(trajs, members):
                placed.append(({f: t[f][0] for f in FRAMES}, s_ / 2))
            bodies.append((body, ctr[F0], rad))
            return trajs, dict(v=v.tolist(), axis=axis.tolist(), angle=float(ang), pivot=pivot.tolist(),
                               analytic_sep_px=float("nan"), sep_target_met=True)
        if far:
            continue
        if (FLAGS.grounded or all(c[2] > rad * 0.6 for c in ctr.values())) and visible(ctr) and clear_of_others(ctr, rad):
            body = (v, axis, ang, pivot)
            sep = body_sep(body, ctr[F0], rad)
            tries = getattr(kinematics, "tries", 0) + 1; kinematics.tries = tries
            if SEP_CHECK and sep < FLAGS.min_sep_px and tries < 400:
                continue
            kinematics.tries = 0
            placed.append((ctr, rad)); bodies.append((body, ctr[F0], rad))
            return trajs, dict(v=v.tolist(), axis=axis.tolist(), angle=float(ang), pivot=pivot.tolist(),
                               analytic_sep_px=sep, sep_target_met=bool(sep >= FLAGS.min_sep_px))
    raise RuntimeError("could not place moving body")


# ------------------------------------------------------------------ analytic two-view separation
Kpx = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
Kinv = np.linalg.inv(Kpx)

def cam_ext(f):
    pos, r, u, fw = cams[f]
    R = np.stack([r, -u, fw]); return R, -R @ pos

def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

def F_from(Rrel, trel):
    return Kinv.T @ skew(trel) @ Rrel @ Kinv

def world_map(body, fa, fb):
    """Rigid world map X(fb) = A X(fa) + d for a moving body (None = static world)."""
    if body is None:
        return np.eye(3), np.zeros(3)
    v, axis, ang, pivot = body
    Ra = quat_to_R(quat_axis_angle(axis, ang * TAU[fa])); Rb = quat_to_R(quat_axis_angle(axis, ang * TAU[fb]))
    A = Rb @ Ra.T
    ca, cb = pivot + v * TAU[fa], pivot + v * TAU[fb]
    return A, cb - A @ ca

def F_body(body, fa, fb):
    Ra, ta = cam_ext(fa); Rb, tb = cam_ext(fb)
    A, d = world_map(body, fa, fb)
    Rrel = Rb @ A @ Ra.T
    trel = Rb @ (d - A @ Ra.T @ ta) + tb
    return F_from(Rrel, trel)

def proj_px(X, f):
    R, t = cam_ext(f); P = X @ R.T + t
    return P[:, :2] / P[:, 2:3] @ np.diag([fx, fx]) + np.array([W / 2, H / 2]), P[:, 2]

def sampson_px(F, x1, x2):
    a = np.c_[x1, np.ones(len(x1))]; b = np.c_[x2, np.ones(len(x2))]
    Fx1 = a @ F.T; Ftx2 = b @ F
    num = (b * Fx1).sum(1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))

def body_points(body, center0, rad, n=300):
    X = center0 + rng.normal(size=(n, 3)) * rad / 2
    A, d = world_map(body, F0, F1)
    return X, X @ A.T + d

def background_points(n=600):
    X = np.c_[rng.uniform(-6, 6, (n, 2)), np.zeros(n)]
    x1, z1 = proj_px(X, F0); x2, z2 = proj_px(X, F1)
    ok = (z1 > 0) & (z2 > 0) & (x1[:, 0] > 0) & (x1[:, 0] < W) & (x1[:, 1] > 0) & (x1[:, 1] < H) \
         & (x2[:, 0] > 0) & (x2[:, 0] < W) & (x2[:, 1] > 0) & (x2[:, 1] < H)
    return X[ok], X[ok]

bodies = [(None, None, None)]  # (body params, center0, rad); index 0 = static world

def body_sep(body, center0, rad):
    """min over existing bodies of the median Sampson residual, both directions (px)."""
    Xa, Xb = body_points(body, center0, rad)
    x1, _ = proj_px(Xa, F0); x2, _ = proj_px(Xb, F1)
    Fb = F_body(body, F0, F1)
    seps = []
    for (ob, oc, orad) in bodies:
        Fo = F_body(ob, F0, F1)
        if ob is None:
            Ya, Yb = background_points()
        else:
            Ya, Yb = body_points(ob, oc, orad)
        y1, _ = proj_px(Ya, F0); y2, _ = proj_px(Yb, F1)
        seps.append(min(np.median(sampson_px(Fo, x1, x2)), np.median(sampson_px(Fb, y1, y2))))
    return float(min(seps))

SEP_CHECK = split not in ("epi_aligned", "camrot")

objects = []  # dicts for metadata

def add(obj, asset_id, size, group, planar=False):
    scene.add(obj)
    objects.append(dict(obj=obj, asset_id=asset_id, size=float(size), group=group, planar=planar))

def add_moving(members_spec, group, kind="generic"):
    kinematics.lifts = []
    members = [(o, s, off) for (o, a, s, off, pl) in members_spec]
    trajs, params = kinematics(members, kind)
    for (o, a, s, off, pl), tr in zip(members_spec, trajs):
        add(o, a, s, group, planar=pl)
        for f in FRAMES:
            o.position, o.quaternion = tr[f]
            o.keyframe_insert("position", f); o.keyframe_insert("quaternion", f)
    return params

motions = {}
if split in ("n2_generic", "n3_generic", "camrot", "planar", "epi_aligned"):
    n_mov = 2 if split == "n3_generic" else 1
    for g in range(1, n_mov + 1):
        if split == "planar":
            o, a, s = new_card(); spec = [(o, a, s, np.zeros(3), True)]
        else:
            o, a, s = new_gso(size=None if FLAGS.grounded else rng.uniform(1.8, 2.6)); spec = [(o, a, s, np.zeros(3), False)]
        motions[g] = add_moving(spec, g, "epi" if split == "epi_aligned" else "generic")
        if split == "planar":
            texture_card(o)
elif split == "confound_same":
    o, a, s = new_gso(size=None if FLAGS.grounded else rng.uniform(1.8, 2.6))
    motions[1] = add_moving([(o, a, s, np.zeros(3), False)], 1)
    o2, _, _ = new_gso(asset_id=a, size=s)
    if place_static(o2, s):
        add(o2, a, s, 0)
elif split == "confound_glued":
    gs = (lambda: None) if FLAGS.grounded else (lambda: rng.uniform(1.4, 1.9))
    oA, aA, sA = new_gso(size=gs()); oB, aB, sB = new_gso(size=gs())
    off = rand_unit(rng); off[2] = 0; off = off / np.linalg.norm(off) * 0.5 * (sA + sB) * 0.55
    motions[1] = add_moving([(oA, aA, sA, off, False), (oB, aB, sB, -off, False)], 1)

elif split == "confound_farglued":
    oA, aA, sA = new_gso(); oB, aB, sB = new_gso()
    dist = rng.uniform(FLAGS.far_lo, FLAGS.far_hi)
    off = rand_unit(rng); off[2] = 0; off = off / np.linalg.norm(off) * dist / 2
    motions[1] = add_moving([(oA, aA, sA, off, False), (oB, aB, sB, -off, False)], 1)
    kinematics.lifts = []
    oC, _, _ = new_gso(asset_id=aA, size=sA)  # identical-looking copy of A, independent motion
    motions[2] = add_moving([(oC, aA, sA, np.zeros(3), False)], 2)

n_static = {"n2_generic": 3, "n3_generic": 2, "camrot": 3, "epi_aligned": 3, "planar": 2,
            "confound_same": 2, "confound_glued": 2, "confound_farglued": 1}[split]
for _ in range(n_static):
    o, a, s = new_gso()
    if place_static(o, s):
        add(o, a, s, 0)

# ------------------------------------------------------------------ render
if FLAGS.no_shadows:
    n_off = 0
    for o in objects:
        bo = o["obj"].linked_objects.get(renderer) if hasattr(o["obj"], "linked_objects") else None
        if bo is None:
            continue
        try:
            bo.cycles_visibility.shadow = False; n_off += 1
        except AttributeError:
            bo.visible_shadow = False; n_off += 1
    logging.info("shadows disabled on %d objects", n_off)

data = renderer.render(return_layers=("rgba", "depth", "segmentation", "forward_flow"))
kb.write_image_dict({"rgba": data["rgba"]}, output_dir)
np.savez_compressed(output_dir / "gt.npz", depth=data["depth"].astype(np.float32),
                    segmentation=data["segmentation"].astype(np.uint8),
                    forward_flow=data["forward_flow"].astype(np.float32))

def mw(asset, f):
    with asset.at_frame(f):
        return np.array(asset.matrix_world).tolist()

meta = dict(split=split, seed=FLAGS.seed, grounded=int(FLAGS.grounded), float_height=FLAGS.float_height,
            no_shadows=int(FLAGS.no_shadows), size_range=[FLAGS.size_lo, FLAGS.size_hi], resolution=[W, H], frames=FRAMES,
            focal_length=FOCAL, sensor_width=SENSOR,
            K=[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
            camera_convention="blender (camera looks along -Z, +Y up); depth = distance along ray",
            camera_matrix_world={f: mw(scene.camera, f) for f in FRAMES},
            background_hdri=hdri_id,
            # renderer writes segmentation value = index in scene.assets + 1 (verified on smoke scenes)
            seg_index_map={str(i + 1): ("dome" if a is dome else [k for k, o in enumerate(objects) if o["obj"] is a][0])
                           for i, a in enumerate(scene.assets) if a is dome or any(o["obj"] is a for o in objects)}, motions={str(k): v for k, v in motions.items()},
            objects=[dict(asset_id=o["asset_id"], size=o["size"], group=o["group"], planar=o["planar"],
                          segmentation_id=int(getattr(o["obj"], "segmentation_id", None) or -1),
                          name=o["obj"].name,
                          foreground_index=scene.foreground_assets.index(o["obj"]) + 1
                          if o["obj"] in scene.foreground_assets else -1,
                          matrix_world={f: mw(o["obj"], f) for f in FRAMES}) for o in objects])
with open(output_dir / "e5_meta.json", "w") as fh:
    json.dump(meta, fh)
kb.done()
