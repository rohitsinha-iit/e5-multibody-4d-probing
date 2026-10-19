set +e
D=${E5_ROOT:-$HOME/e5_multibody}/data/v2; JW=$PWD; mkdir -p $D
cp kubric_e5_scene.py e5_kubric_post.py e5_post_driver.py e5_multibody.py render_v2_pilot.sh $D/
cd $D && bash render_v2_pilot.sh || { cp logs/smoke_v2.txt $JW/; exit 1; }
echo "scenes $(ls out | wc -l) meta $(ls out/*/e5_meta.json | wc -l)"; grep -l Traceback logs/*.txt || true
${E5_ROOT:-$HOME/e5_multibody}/.venv-post/bin/python e5_post_driver.py out > post.log 2>&1; tail -3 post.log
mkdir -p $JW/sheets; cp sheets/* $JW/sheets/; cp scene_summary.json $JW/
