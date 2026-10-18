# Item 1 at scale (after approval of the re-rendered pilot): 20 far-glued scenes.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh
step "rendering far-glued set (resumable)"; render_list ALL "v3far"
step "post v3far"; post_ds $(ds_dir v3far)
mkdir -p $RESULT_DIR/v3far; cp $(ds_dir v3far)/scene_summary.json $RESULT_DIR/v3far/
echo "v3far scenes: $(ls $(ds_dir v3far)/out | wc -l)"
