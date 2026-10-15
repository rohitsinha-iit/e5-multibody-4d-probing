# Re-render the far-glued pilot with the image-space gap constraint (coupled members >= 0.22 W apart in every frame).
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh
set -e
D=$(ds_dir v3far)
rm -rf $D/out/confound_farglued_s501 $D/out/confound_farglued_s502 $D/out/confound_farglued_s503 $D/sheets
render_list P
post_ds $D
rm -rf $RESULT_DIR/v3far; mkdir -p $RESULT_DIR/v3far; cp $D/sheets/confound_farglued_* $D/scene_summary.json $RESULT_DIR/v3far/
ls $RESULT_DIR/v3far
