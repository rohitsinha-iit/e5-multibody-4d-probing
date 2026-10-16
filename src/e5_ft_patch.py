"""Item 10 helper (idempotent). Usage: python e5_ft_patch.py SRC_DIR SCRIPTS_DIR
1. Adds a backward-compatible --checkpoint argument to SRC_DIR/e5_realdata_run.py (empty = the model's default
   checkpoint, so existing tasks are unchanged) and copies the file to SCRIPTS_DIR.
2. Writes SCRIPTS_DIR/e5_ft_extract.py: a copy of e5_any4d_extract.py whose checkpoint path is read from $E5_CKPT
   (default: the base checkpoint). The shared extract script itself is not modified.
"""
import os, sys
src, scr = sys.argv[1], sys.argv[2]
p = os.path.join(src, "e5_realdata_run.py"); t = open(p).read()
if '"--checkpoint"' not in t:
    a1 = 'ap.add_argument("--limit", type=int, default=0)'
    a2 = 'model = e5_models.get_model(a.model, resize_mode=mode)'
    assert a1 in t and a2 in t, "e5_realdata_run.py changed; cannot add --checkpoint"
    t = t.replace(a1, 'ap.add_argument("--checkpoint", default="", help="Any4D checkpoint path (default: model default)")\n' + a1, 1)
    t = t.replace(a2, 'model = e5_models.get_model(a.model, resize_mode=mode, **({"checkpoint_path": a.checkpoint} if a.checkpoint else {}))', 1)
    open(p + ".tmp", "w").write(t); os.replace(p + ".tmp", p); print("patched", p)
open(os.path.join(scr, "e5_realdata_run.py"), "w").write(t)
e = open(os.path.join(src, "e5_any4d_extract.py")).read()
old = '"checkpoint_path": "checkpoints/any4d_4v_combined.pth"'
assert old in e, "e5_any4d_extract.py changed; cannot derive e5_ft_extract.py"
e = e.replace(old, '"checkpoint_path": os.environ.get("E5_CKPT", "checkpoints/any4d_4v_combined.pth")', 1)
open(os.path.join(scr, "e5_ft_extract.py"), "w").write("# derived from e5_any4d_extract.py by e5_ft_patch.py (checkpoint from $E5_CKPT)\n" + e)
print("ok")
