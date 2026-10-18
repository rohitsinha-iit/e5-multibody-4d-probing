# Idempotent environment setup for the cross-model track (item 7). Sourced by the t7_* task scripts; safe to rerun.
# Everything lives on the big disk: venv, source clones, Hugging Face cache, uv/pip caches.
XM=$BIG/xmodel; XMV=$BIG/models/venvs/xm; XMS=$BIG/models/src
export HF_HOME=$BIG/models/hf UV_CACHE_DIR=$BIG/models/uvcache PIP_CACHE_DIR=$BIG/models/pipcache TMPDIR=$BIG/models/tmp
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p $XM $XMS $HF_HOME $UV_CACHE_DIR $TMPDIR
xm_setup() {
  local UV=$HOME/.local/bin/uv
  if [ ! -f $XMV/.ok ]; then
    step "creating venv $XMV"
    $UV venv -q --python 3.12 $XMV
    VIRTUAL_ENV=$XMV $UV pip install -q torch==2.6.0 torchvision==0.21.0 numpy scipy scikit-learn pillow huggingface_hub safetensors einops matplotlib
    test -d $XMS/vggt/.git || git clone -q https://github.com/facebookresearch/vggt $XMS/vggt
    test -d $XMS/map-anything/.git || git clone -q https://github.com/facebookresearch/map-anything $XMS/map-anything
    test -d $XMS/PAGE4D/.git || git clone -q https://github.com/kaichen-z/PAGE4D $XMS/PAGE4D
    VIRTUAL_ENV=$XMV $UV pip install -q --no-deps -e $XMS/vggt
    VIRTUAL_ENV=$XMV $UV pip install -q -e $XMS/map-anything
    VIRTUAL_ENV=$XMV $UV pip install -q torch==2.6.0 torchvision==0.21.0   # keep torch pinned if map-anything pulled another
    $XMV/bin/python -c "import torch, vggt, mapanything; print('torch', torch.__version__, torch.cuda.is_available())" && touch $XMV/.ok
  fi
  test -f $XMV/.ok || { echo "xm venv setup failed"; exit 3; }
}
xm_weights() {  # public, non-gated weights only; no token is used
  $XMV/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
for repo, fn, kw in [("facebook/VGGT-1B", "model.pt", {}), ("facebook/VGGT_tracker_fixed", "model_tracker_fixed_e20.pt", {}),
                     ("zhouk777/PAGE4D", "checkpoint_nomask.pt", {"repo_type": "dataset"})]:
    print(hf_hub_download(repo, fn, **kw), flush=True)
print(snapshot_download("facebook/map-anything", revision="6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a"), flush=True)  # V1: dinov2_large encoder (input_embed_dim 1024) + aat_24_layers_ifr (depth 24), the same config names as Any4D (main is the larger V1.1, dinov2-giant, 16 blocks)
EOF
}
xm_env() { source $XMV/bin/activate; export CUDA_VISIBLE_DEVICES=${XM_GPU:-$(pick_gpu)}; echo "GPU $CUDA_VISIBLE_DEVICES"; }
