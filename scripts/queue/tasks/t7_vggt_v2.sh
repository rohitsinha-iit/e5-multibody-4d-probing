# Item 7 cross-model: VGGT-1B (static) on v2; tracks from the public facebook/VGGT_tracker_fixed head (VGGT-1B's own head gives 28 to 93 px static EPE on smoke scenes): outputs (3D/2D slope, sep_ratio), probes P, attention A.
MODEL=vggt RUNNER=e5_xm_vggt.py CKPAT="VGGT-1B.*model.pt" TRKPAT="VGGT_tracker_fixed.*e20.pt" XLAY=0,8,16,24,32,40,48
source $QDIR/tasks/t7_xm_model.sh
