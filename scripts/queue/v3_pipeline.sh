# Run the pipeline on the new datasets: far-glued (C3), floating (C4), and v2 with 20 extra epi scenes (C4).
source $QDIR/tasks/common.sh; set -e
EVEN=$(seq -s, 0 2 24),17,19,21
bash $QDIR/tasks/pipeline.sh v3far "$EVEN" v3far
bash $QDIR/tasks/pipeline.sh v3float "$EVEN" v3float
bash $QDIR/tasks/pipeline.sh v2 "" v2_120
