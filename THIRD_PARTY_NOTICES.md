# Third-party notices

BeyondRetarget Plus is released under CC BY-NC 4.0 for the original project
code and documentation only. The components below are not relicensed by this
repository. Verify the upstream terms before downloading, modifying, or
redistributing any checkpoint, dataset, model file, or robot asset.

| Component | Use in Plus | Upstream | License / redistribution status |
| --- | --- | --- | --- |
| TRAM | human trajectory, camera, dataset, and training components | [github.com/yufu-wang/tram](https://github.com/yufu-wang/tram) | MIT; notice retained in `thirdparty/tram/LICENSE` |
| 4D-Humans / HMR2 | frozen human feature extractor and VIMO backbone | [github.com/shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | Follow upstream code and checkpoint terms |
| OpenMMLab / ViTDet | optional camera frontend | [github.com/open-mmlab](https://github.com/open-mmlab) | Apache-2.0 for the relevant code; weights have separate terms |
| DROID-SLAM | masked metric camera tracking | [github.com/princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | See `thirdparty/DROID-SLAM/LICENSE` |
| ZoeDepth | optional depth/scale estimation | [github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) | See `thirdparty/ZoeDepth/LICENSE` |
| Segment Anything | optional mask generation | [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 code; checkpoint terms apply |
| DEVA | optional tracking frontend | [github.com/hkchengrex/Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) | Obtain and install from upstream under its terms |
| SPEC/camcalib | camera FOV/gravity calibration for moving-camera preprocessing | `thirdparty/camcalib/` and upstream TRAM resource links | Source headers retain the Max-Planck proprietary notice; use only under a valid upstream license; checkpoint is downloaded separately |
| PARE/TRAM Pascal occlusion helper | optional training-time synthetic occlusion | `lib/datasets/coco_occlusion.py`, adapted from [isarandi/synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion) | Upstream MPG notice is retained; Pascal-derived resource is downloaded separately and remains subject to its own terms |
| PyTorch3D | geometry and rotation operators | [github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | BSD-style upstream license |
| Ultralytics YOLO | person-box and COCO17 pose extraction | [ultralytics.com/license](https://www.ultralytics.com/license) | AGPL-3.0 or commercial terms; do not redistribute weights without compliance |
| SMPL / SMPLify | body model and fitting resources | [SMPL](https://smpl.is.tue.mpg.de), [SMPLify](https://smplify.is.tue.mpg.de) | Registration/gated terms; users download directly |
| EMDB, BEDLAM, H36M, 3DPW | training and evaluation data | Official dataset providers | Dataset-specific terms; not included here |
| Robot URDF/MJCF and meshes | forward kinematics and visualization | Individual robot providers | Provider-specific terms; redistribute `robot.zip` only after permission |

## Components with separate terms

The SPEC/camcalib source is included in `thirdparty/camcalib/` in the same
layout used by TRAM. Every upstream Max-Planck header is retained. The source
is not relicensed by this repository and requires a valid upstream license;
the checkpoint is not committed and must be obtained separately from the
official TRAM resource after accepting those terms; `scripts/download_models.sh`
intentionally does not download it.

The synthetic occlusion helper inherited from the PARE/TRAM training stack is
included in `lib/datasets/coco_occlusion.py` with its original MPG notice and
the adaptation reference to `isarandi/synthetic-occlusion`. The Pascal-derived
occluder resource is not committed; download it with
`scripts/download_pascal_occluders.sh` and comply with its upstream terms.
Public Plus configs keep `DATASET.OCCLUSION=false` by default, but the original
augmentation remains available when the resource is installed.

The Plus asset bundle must not include gated SMPL files, datasets, or model
weights unless the relevant upstream provider explicitly permits redistribution.
