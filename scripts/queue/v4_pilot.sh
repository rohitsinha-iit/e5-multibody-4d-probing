# Review item W3 pilot: 2 scenes per new split (epi_accel, epi_const, epi_sizerand, epi_sizematch) for visual approval.
source $QDIR/tasks/common.sh; source $QDIR/tasks/common_extra.sh; source $QDIR/tasks/v4_common.sh
set -e
step "smoke renders (128 px, 4 frames)"; v4_smoke_renders
step "pilot renders"; v4_render_list P
D=$(ds_dir $V4DS)
step "post (flow check, SNR, contact sheets, GIFs)"; post_ds $D
cp $SRC/e5_v4_epicheck.py $SRC/e5_step3_analysis.py $D/
step "GT checks"; (cd $D && $POST e5_v4_epicheck.py out $RESULT_DIR/v4_pilot "v4 pilot: GT checks of the new epipolar scene types" > epicheck_pilot.log 2>&1; tail -2 epicheck_pilot.log)
mv $RESULT_DIR/v4_pilot_summary.md $RESULT_DIR/v4_pilot_summary.md.tmp
{ echo "# v4_pilot: pilot of the new epipolar scene types (review item W3)"; echo
  echo "Two scenes per new split, rendered with the v3float settings (floating, shadowless, sizes 1.4 to 2.2 m for static objects)."
  echo "Contact sheets and GIFs are in sheets/. The scaled task v4_scale_epi waits for the gate approvals/v4_visual."; echo
  echo "What to check visually:"
  echo "- epi_accel vs epi_const with the same seed: frames 0 and 23 look the same; in the GIF the accelerated object starts or stops while the constant one glides."
  echo "- epi_sizerand: the moving object can be much smaller or larger than the static objects."
  echo "- epi_sizematch: the moving object is large (2.2 to 4.4 m), so read as a static object at the closer ambiguous depth it would be of typical size."
  echo "- Nothing touches the floor, no shadows, every object stays in view."; echo
  tail -n +2 $RESULT_DIR/v4_pilot_summary.md.tmp; } > $RESULT_DIR/v4_pilot_summary.md
rm -f $RESULT_DIR/v4_pilot_summary.md.tmp
mkdir -p $RESULT_DIR/sheets; cp $D/sheets/* $RESULT_DIR/sheets/; cp $D/scene_summary.json $RESULT_DIR/
ls $RESULT_DIR $RESULT_DIR/sheets | head -40
