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
| SPEC/camcalib | camera FOV/gravity calibration | [TRAM-compatible source](https://github.com/yufu-wang/tram/tree/main/thirdparty/camcalib), [SPEC terms](https://github.com/mkocabas/SPEC/blob/master/LICENSE) | Source and checkpoint are not included; users must obtain them under the upstream license |
| PARE/TRAM Pascal occlusion helper | optional training-time synthetic occlusion | [TRAM-compatible source](https://github.com/yufu-wang/tram/blob/main/lib/datasets/coco_occlusion.py), [PARE terms](https://github.com/mkocabas/PARE/blob/master/LICENSE), [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion) | Restricted implementation is not included; source and Pascal resource must be obtained under their respective upstream terms |
| PyTorch3D | geometry and rotation operators | [github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | BSD-style upstream license |
| Ultralytics YOLO | person-box and COCO17 pose extraction | [ultralytics.com/license](https://www.ultralytics.com/license) | AGPL-3.0 or commercial terms; do not redistribute weights without compliance |
| SMPL / SMPLify | body model and fitting resources | [SMPL](https://smpl.is.tue.mpg.de), [SMPLify](https://smplify.is.tue.mpg.de) | Registration/gated terms; users download directly |
| EMDB, BEDLAM, H36M, 3DPW | training and evaluation data | Official dataset providers | Dataset-specific terms; not included here |
| Robot URDF/MJCF and meshes | forward kinematics and visualization | Individual robot providers | Provider-specific terms; redistribute `robot.zip` only after permission |

## Components with separate terms

This project does not have permission to redistribute the SPEC/camcalib or
PARE/TRAM occlusion implementation. Neither is included or automatically
downloaded. Users must obtain compatible copies under the applicable upstream
licenses, retain their notices, and place them at the fixed paths in
[README.md](README.md#spec-and-pare-components). Upstream links identify the
components; this project does not grant rights to those sources or resources.

SPEC source and its checkpoint are required to generate gravity-aligned
world-camera results in all camera modes. The optional occlusion adapter in
`lib/datasets/coco_occlusion.py` only forwards calls to user-supplied code at
`thirdparty/pare/coco_occlusion.py`; it contains no upstream implementation.
Public Plus configs keep `DATASET.OCCLUSION=false` by default. The separate
Pascal resource can be obtained with `scripts/download_pascal_occluders.sh`
under its upstream terms.

The Plus asset bundle must not include gated SMPL files, datasets, or model
weights unless the relevant upstream provider explicitly permits redistribution.
