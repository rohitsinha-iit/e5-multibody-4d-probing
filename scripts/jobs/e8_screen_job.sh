set -e
export PATH=$HOME/.local/bin:$PATH; W=$PWD; D=${E5_ROOT:-$HOME/e5_multibody}/data/pilot_v1
cd ${E5_ROOT:-$HOME/e5_multibody}/Any4D && source .venv/bin/activate && cp $W/e5_e8_knockout.py scripts/
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); echo GPU $GPU
A="--data $D/out --gt $D/gt_patches --att $D/internals --e5 $D"
CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_e8_knockout.py $A --screen 1 --per_split 3 --out $D/e8_screen.json > $D/e8_screen.log 2>&1; cp $D/e8_screen.json $D/e8_screen.log $W/
grep -v -i warn $D/e8_screen.log | tail -3
