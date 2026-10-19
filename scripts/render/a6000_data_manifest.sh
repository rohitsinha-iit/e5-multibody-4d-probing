#!/bin/bash
# Writes env/a6000_data.tsv: every E5 dataset on the A6000 (path, scene/file count, size, generator sha256, flags, status).
# Usage: bash scripts/render/a6000_data_manifest.sh   (safe to re-run; reads only)
E5_BIG=${E5_BIG:-/data2/DATA/rohit_sinha/e5/big}; REPO=$(cd $(dirname $0)/../.. && pwd); OUT=$REPO/env/a6000_data.tsv
sha() { sha256sum $1 | cut -c1-12; }
GN=$(sha $REPO/src/attr/kubric_e5_scene_native.py); GV4=$(sha $REPO/src/attr/kubric_e5_scene_v4_native.py); GF=$(sha $REPO/src/attr/kubric_e5_scene_factorial.py)
n_meta() { ls -d $1/*/ 2>/dev/null | while read d; do [ -f $d/e5_meta.json ] && echo x; done | wc -l; }
sz() { du -shL $1 2>/dev/null | cut -f1; }
{
printf "# E5 data on the A6000 (aicoe-a6000), written %s by scripts/render/a6000_data_manifest.sh\n" "$(date '+%F %T')"
printf "# render settings for all Kubric sets: 512x512, 24 frames (frame_end 24), 32 spp, --min_sep_px 25; native bpy 4.3.0 + kubric 61f2422 (h4-kubric env)\n"
printf "dataset\tpath\tcount\texpected\tsize\tgenerator (sha256 prefix of file run; original sha)\tflags\tnotes\n"
printf "v2 (D)\t$E5_BIG/data/v2/out\t%s\t100\t%s\tsrc/attr/kubric_e5_scene_native.py $GN (port of f65cfd0b)\t--grounded 1 --size_lo 1.4 --size_hi 2.2\tout100 = s301-s314 + s401-s486; out100 -> out symlink\n" $(n_meta $E5_BIG/data/v2/out) $(sz $E5_BIG/data/v2/out)
printf "v3float (D)\t$E5_BIG/data/v3float/out\t%s\t70\t%s\tsrc/attr/kubric_e5_scene_native.py $GN (port of f65cfd0b)\t--grounded 1 --size_lo 1.4 --size_hi 2.2 --float_height 0.6 --no_shadows 1\ts531-s600\n" $(n_meta $E5_BIG/data/v3float/out) $(sz $E5_BIG/data/v3float/out)
printf "v4epi (D)\t$E5_BIG/data/v4epi/out\t%s\t60\t%s\tsrc/attr/kubric_e5_scene_v4_native.py $GV4 (port of c6e4d25a)\t--grounded 1 --size_lo 1.4 --size_hi 2.2 --float_height 0.6 --no_shadows 1\tepi_accel + epi_const s801-s830 (round 2 rows only; epi_sizerand/epi_sizematch not rendered)\n" $(n_meta $E5_BIG/data/v4epi/out) $(sz $E5_BIG/data/v4epi/out)
for ds in v2 v3float; do for v in A B C; do
  n=$(ls -d $E5_BIG/attr/factorial/$ds/*/$v 2>/dev/null | while read d; do [ -f $d/e5_meta.json ] && echo x; done | wc -l)
  case $v in A) f="--freeze_camera 1 --freeze_objects 1";; B) f="--freeze_objects 1";; C) f="--freeze_camera 1";; esac
  e=$([ $ds = v2 ] && echo 100 || echo 70)
  printf "factorial $ds $v\t$E5_BIG/attr/factorial/$ds/<seed>/$v\t$n\t$e\t%s\tsrc/attr/kubric_e5_scene_factorial.py $GF (from f65cfd0b)\t$f + set flags\tframe 0 RGB = byte copy of D (rendered kept as rgba_00000_rendered.png); gt_factorial.npz per variant; D = symlink, D_gt holds D's gt_factorial.npz\n" "$(du -shc $E5_BIG/attr/factorial/$ds/*/$v 2>/dev/null | tail -1 | cut -f1)"
done; done
for k in kitti15 sintel; do printf "realdata $k\t$E5_BIG/realdata/$k\t%s files\t-\t%s\tsrc/e5_realdata_fetch.sh\t-\tGT: $E5_BIG/realdata/gt/$k (%s npz)\n" "$(find $E5_BIG/realdata/$k -type f 2>/dev/null | wc -l)" "$(sz $E5_BIG/realdata/$k)" "$(ls $E5_BIG/realdata/gt/$k 2>/dev/null | wc -l)"; done
for p in sintel tum bonn davis; do printf "poseset $p\t$E5_BIG/realdata/posesets/$p\t%s files\t-\t%s\tHF yslan/pointmap_regression_evalsets $p.tar.gz\t-\textracted=%s; tarball kept in posesets/_tar\n" "$(find $E5_BIG/realdata/posesets/$p -type f 2>/dev/null | wc -l)" "$(sz $E5_BIG/realdata/posesets/$p)" "$([ -f $E5_BIG/realdata/posesets/.done_$p ] && echo yes || echo no)"; done
printf "cotracker3_kubric\t$E5_BIG/data/cotracker3_kubric\t%s shards done\t400\t%s\tHF facebook/CoTracker3_Kubric (downloaded by the cause track)\t-\teach shard = one 120-frame sequence, extracted to <shard>/ with <shard>.done\n" "$(ls $E5_BIG/data/cotracker3_kubric/*.done 2>/dev/null | wc -l)" "$(sz $E5_BIG/data/cotracker3_kubric)"
} > $OUT
cat $OUT
