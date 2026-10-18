# Item 10 rerun: gain with checkpoint selection on scene-flow val loss and a static-EPE guard
FT_RUN_TAG=_sel FT_TRAIN_ARGS="--select sf --static_tol 0.25" bash $QDIR/tasks/ft_train.sh gain
