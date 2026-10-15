# Steps 3-6 + adjacency control on the floating set and on v2 with 30 epi scenes.
source $QDIR/tasks/common.sh; set -e
EVEN=$(seq -s, 0 2 24),17,19,21
bash $QDIR/tasks/pipeline.sh v3float "$EVEN" v3float
bash $QDIR/tasks/pipeline.sh v2 "" v2_120
