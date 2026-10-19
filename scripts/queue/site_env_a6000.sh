# A6000 (aicoe-a6000) site settings for E5. Source before starting the supervisor or running a task by hand:
#   . /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
# common.sh sources $QDIR/site_env.sh automatically when E5_SITE_ENV is not yet set.
export E5_SITE_ENV=a6000
export E5_ROOT=/data2/DATA/rohit_sinha/e5
export E5_BIG=/data2/DATA/rohit_sinha/e5/big
export E5_QUEUE=$E5_BIG/queue
export PIP_CACHE_DIR=$E5_BIG/cache/pip UV_CACHE_DIR=$E5_BIG/cache/uv XDG_CACHE_HOME=$E5_BIG/cache/xdg
export HF_HOME=$E5_BIG/models/hf TORCH_HOME=$E5_BIG/cache/torch HF_HUB_DISABLE_TELEMETRY=1
export PATH=$E5_BIG/tools/uvpkg/bin:$PATH
# shared box: at most 12 CPU threads per task, 4 GB GPU headroom on top of each task's GPU_NEED_GB
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-12} NUMEXPR_MAX_THREADS=12
export GPU_HEADROOM_GB=${GPU_HEADROOM_GB:-4}
export E5_MAX_PARALLEL=${E5_MAX_PARALLEL:-2}
