set -e
export PATH=$HOME/.local/bin:$PATH; W=$PWD; D=~/e5_multibody/data/pilot_v1
cd ~/e5_multibody/Any4D && source .venv/bin/activate && cp $W/e5_e8_knockout.py scripts/
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); echo GPU $GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
A="--data $D/out --gt $D/gt_patches --att $D/internals --e5 $D"
CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_e8_knockout.py $A --out $W/e8_test.json --limit 2 2>&1 | grep -v -i warn | tail -4
CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_e8_knockout.py $A --out $W/e8_knockout.json > $W/e8.log 2>&1
grep -v -i warn $W/e8.log | tail -3; cp $W/e8_knockout.json $D/
