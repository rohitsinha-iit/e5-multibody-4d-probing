# Round 2, review item 1 (W2) shared setup. Sourced after common.sh. Idempotent.
# Uses the existing xm venv (torch 2.6, torchvision 0.21, huggingface_hub, einops) without installing anything into it:
# CoTracker3 runs from a source clone on sys.path, RAFT from torchvision. Public, non-gated weights only, no token.
XM=$BIG/xmodel; XMV=$BIG/models/venvs/xm; W2=${W2_ROOT:-$BIG/w2}
export HF_HOME=$BIG/models/hf TORCH_HOME=$BIG/models/torch TMPDIR=$BIG/models/tmp HF_HUB_DISABLE_TELEMETRY=1
mkdir -p $W2/pred $W2/eval $TORCH_HOME $TMPDIR
CTS=$BIG/models/src/co-tracker; CT_COMMIT=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
V3F=$BIG/data/v3float/out; RD=$BIG/realdata
A4PY=$A4/.venv/bin/python          # CPU evaluation only (numpy, scipy, cv2, matplotlib); nothing is installed into it
W2LIM=${W2_LIMIT:+--limit $W2_LIMIT}
ds_path() { case $1 in v2) echo $V2/out100;; v3float) echo $V3F;; esac; }
w2_setup() {
  test -f $XMV/.ok || { echo "xm venv missing at $XMV"; exit 3; }
  test -d $CTS/.git || git clone -q https://github.com/facebookresearch/co-tracker $CTS
  (cd $CTS && git checkout -q $CT_COMMIT)
  CTCK=$($XMV/bin/python -c "from huggingface_hub import hf_hub_download as h; print(h('facebook/cotracker3', 'scaled_offline.pth'))" 2>/dev/null | tail -1)
  test -f "$CTCK" || { echo "CoTracker3 weights missing"; exit 4; }
  $XMV/bin/python -c "from torchvision.models.optical_flow import raft_large, Raft_Large_Weights as W; raft_large(weights=W.C_T_V2, progress=False)" \
    || { echo "RAFT weights missing"; exit 4; }
  echo "co-tracker $(cd $CTS && git rev-parse --short HEAD), weights $CTCK"
}
w2_gpu() { export CUDA_VISIBLE_DEVICES=${W2_GPU:-$(pick_gpu)}; echo "GPU $CUDA_VISIBLE_DEVICES"; }
# pred_dir MODEL DATASET: where the e5_xm_eval-format predictions of a model live
pred_dir() {
  case $1/$2 in
    any4d/v2) echo $XM/any4d/pred;; vggt/v2) echo $XM/vggt/pred;;
    any4d/v3float) echo $W2/pred/any4d_v3float;; vggt/v3float) echo $W2/pred/vggt_v3float/pred;;
    *) echo $W2/pred/$1_$2;;
  esac
}
W2_MODELS="any4d vggt cotracker3_24f cotracker3_7f raft_direct raft_chain7 raft_chain24"
