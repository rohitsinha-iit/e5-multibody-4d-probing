# Item 7 cross-model: PAGE-4D (dynamic VGGT fine-tune) on v2: outputs (3D/2D slope, sep_ratio), probes P, attention A.
MODEL=page4d RUNNER=e5_xm_vggt.py CKPAT="PAGE4D.*checkpoint_nomask.pt" XLAY=0,8,16,24,32,40,48
source $QDIR/tasks/t7_xm_model.sh
