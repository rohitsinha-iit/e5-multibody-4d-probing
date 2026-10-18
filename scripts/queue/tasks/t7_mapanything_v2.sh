# Item 7 cross-model: MapAnything (static) on v2: outputs (3D/2D slope, sep_ratio), probes P, attention A.
MODEL=mapanything RUNNER=e5_xm_mapanything.py CKPAT="map-anything/snapshots/6f3a25bf" XLAY=0,4,8,12,16,20,24
source $QDIR/tasks/t7_xm_model.sh
