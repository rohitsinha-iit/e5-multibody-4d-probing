# Item 2 at scale (approved): 70 floating shadowless scenes + 20 extra grounded epi scenes in v2.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh
step "rendering floating set and extra v2 epi (resumable)"; render_list ALL "v3float|v2"
for ds in v3float v2; do
  step "post $ds"; post_ds $(ds_dir $ds)
  mkdir -p $RESULT_DIR/$ds; cp $(ds_dir $ds)/scene_summary.json $RESULT_DIR/$ds/
done
echo "v3float scenes: $(ls $(ds_dir v3float)/out | wc -l)  v2 scenes: $(ls $V2/out | wc -l)"
