# Steps 3-6 + adjacency control on the far-glued set (C3 adjacency control).
source $QDIR/tasks/common.sh; set -e
EVEN=$(seq -s, 0 2 24),17,19,21
bash $QDIR/tasks/pipeline.sh v3far "$EVEN" v3far
