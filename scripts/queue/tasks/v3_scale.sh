# Items 1+2 at scale (after visual approval): 20 far-glued scenes, 70 floating shadowless scenes, 20 extra grounded epi (v2).
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh
step "rendering all v3 scenes (resumable)"; render_list ALL
for ds in v3far v3float v2; do
  step "post $ds"; post_ds $(ds_dir $ds)
  mkdir -p $RESULT_DIR/$ds; cp $(ds_dir $ds)/scene_summary.json $RESULT_DIR/$ds/
done
for ds in v3far v3float; do echo "$ds scenes: $(ls $(ds_dir $ds)/out | wc -l)"; done; echo "v2 scenes: $(ls $E5/data/v2/out | wc -l)"
