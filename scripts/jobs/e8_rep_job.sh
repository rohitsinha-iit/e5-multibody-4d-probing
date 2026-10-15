set -e
export PATH=$HOME/.local/bin:$PATH; W=$PWD; D=~/e5_multibody/data/v2
cd ~/e5_multibody/Any4D && source .venv/bin/activate && cp $W/e5_e8_knockout.py scripts/
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
A="--data $D/out --gt $D/gt_patches --att $D/internals --e5 $D"
CUDA_VISIBLE_DEVICES=$GPU python scripts/e5_e8_knockout.py $A --heads 2:1,22:9,17:10,14:6,8:11 --per_split 5 --out $D/e8_replicate_v2.json > $D/e8_rep.log 2>&1
cp $D/e8_replicate_v2.json $W/; grep -v -i warn $D/e8_rep.log | tail -2
