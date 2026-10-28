# Runtime assets

Small, non-gated metadata and demo media are kept in this tree. Large or
gated resources must be downloaded separately and must not be committed to the
source repository without permission.

## Robot descriptions

The lightweight robot descriptions currently tracked here are:

```text
robot/
  atlas_v4/atlas_v4.urdf
  g1/g1_mocap_29dof.xml
  r1/r1.xml
  gr1t1/gr1t1_without_hand.urdf
  gr2v3_8_7_dummy_hand/gr2v3_8_7_dummy_hand.urdf
  h1_with_hand_without_hand/h1_with_hand_without_hand.urdf
  t1_serial/t1_serial.urdf
  tienkung/tienkung.urdf
```

The complete meshes must be beside the matching URDF/MJCF files. Install the
shared archive from the [BeyondRetarget Base Google Drive folder](https://drive.google.com/drive/folders/12ySWLbVhF8WEGyLR0Ll09DmZm8rZE0Tg)
and other release assets with:

```bash
python scripts/download_base_shared_assets.py
```

The downloader handles both provider directory names (such as `unitree_g1`)
and the canonical Plus names used by the configs. It recursively fills missing
files without overwriting the canonical XML/URDF files tracked in this tree.

Before redistributing that archive, confirm the license and redistribution
rights for every robot provider. The project CC BY-NC license does not replace
those terms.

## Checkpoints and model files

Core Plus inference normally needs:

```text
data/smpl/SMPL_NEUTRAL.pkl              # gated SMPL model file
assets/yolo/yolov8x.pt                  # only for the included box extractor
assets/yolo/yolov8x-pose.pt             # only for the included COCO17 extractor
assets/checkpoints/beyondretarget_plus_8robots.pth
                                        # complete VIMO backbone + eight heads + static head
```

The shared Base Google Drive folder also contains `beyondretarget_plus_8robots.pth`, so
the default setup script downloads the three shared assets from the same link.
The pose checkpoint is downloaded separately from the official Ultralytics
release linked in the root README; SPEC/camcalib remains a separately licensed
external download as documented there.
This eight-robot bundle is a complete inference checkpoint. It embeds the
frozen TRAM/VIMO backbone, eight robot adapter heads, and the shared static
head, so Plus inference does not need a separate VIMO checkpoint. It intentionally has no
learned root/root-residual checkpoint: root motion remains the frozen native
TRAM/VIMO output and post-processing uses full C-RC1 reprojection.

## Demo media

`demos/` contains the small public project visualizations used by the root
README. These media are separate from model weights and datasets.
