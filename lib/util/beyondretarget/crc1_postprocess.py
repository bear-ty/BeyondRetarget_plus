"""Image-supported static processing, optionally including C-RC1 reprojection.

The numerical recipe matches the reviewed 2026-10-17 candidate. Original
network probabilities are distinct from optimization/IK constraint weights.
Only a single, explicitly confirmed, continuous shot is accepted. Image
observations are independent of the corrected 3D trajectory (no self-gating).
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch

from lib.robot.robot_spec import model_dof_to_xml_dof
from lib.util.beyondretarget.crc1 import CRC1Config, refine_from_prefilter
from lib.util.beyondretarget.crc1_reprojection import fit_translation
from lib.util.beyondretarget.crc1_image_evidence import estimate_background, evidence_from_pairs
from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.foot_height import (
    RobotFootGeometry, predicted_human_sole_heights, align_root_height,
)
from lib.util.beyondretarget.motion_filter_v2 import balanced_motion_filter

METHOD = 'C_RC1_observation_root_image_support_v1'
POLICY = ('p>0.5 AND reliable background-compensated ankle image motion; '
          'constraint weights >=0.95 are NOT calibrated probabilities; no world-speed veto')


def load_observations(boxes_path, keypoints_path):
    with np.load(boxes_path, allow_pickle=False) as data:
        boxes = {key: data[key] for key in data.files}
    points = np.load(keypoints_path, allow_pickle=False)
    if isinstance(points, np.lib.npyio.NpzFile):
        with points as data:
            if 'keypoints' not in data:
                raise ValueError('Keypoint NPZ needs keypoints=[T,17,3] in full-image pixels')
            points = {key: data[key] for key in data.files}
    else:
        points = {'keypoints': points}
    return boxes, points


def validate_inputs(raw, spec, boxes, points, *, confirm_single_shot, clearance=0.):
    if not confirm_single_shot:
        raise ValueError('C-RC1 requires split shots/target identities; pass --confirm-single-shot')
    if bool(raw.get('postprocess_applied', True)):
        raise ValueError('C-RC1 requires explicitly unprocessed raw; do not filter twice')
    if not bool(raw.get('static_head_trained', False)):
        raise ValueError('Trained static probabilities required')
    if str(raw.get('robot_name')) != spec.name:
        raise ValueError('Robot/spec mismatch')
    ids = np.asarray(raw['frame'])
    n = len(ids)
    fps = float(raw['fps'])
    if n < 1 or ids.ndim != 1 or not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError('Nonempty integer frame IDs required')
    if np.any(np.diff(ids) != 1):
        raise ValueError('Split frame gaps before C-RC1; no implicit truncation/interpolation')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(clearance):
        raise ValueError('Invalid FPS/clearance')
    fields = {'root_pos_world_zup': (n, 3), 'root_quat_world_zup_wxyz': (n, 4),
              'dof': (n, spec.dof), 'pred_pose': (n, 144), 'pred_shape': (n, 10),
              'static_prob6': (n, 6), 'camera_to_world_zup_R': (n, 3, 3),
              'camera_to_world_zup_T': (n, 3)}
    for key, shape in fields.items():
        if key not in raw or np.shape(raw[key]) != shape or not np.isfinite(raw[key]).all():
            raise ValueError('Invalid/missing raw field: ' + key)
    pred_trans = np.asarray(raw['pred_trans'])
    if pred_trans.shape not in ((n, 3), (n, 1, 3)) or not np.isfinite(pred_trans).all():
        raise ValueError('Invalid camera-space human translation')
    prob = np.asarray(raw['static_prob6'])
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError('Expected probabilities in [0,1], not logits/constraint weights')
    if not np.allclose(np.linalg.norm(raw['root_quat_world_zup_wxyz'], axis=1), 1., atol=1e-4):
        raise ValueError('Expected unit wxyz root quaternions')
    if 'joint_names' in raw and list(raw['joint_names']) != list(spec.model_dof_order):
        raise ValueError('Model-order DoF/joint names mismatch')
    rotation = np.asarray(raw['camera_to_world_zup_R'])
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-3)):
        raise ValueError('Invalid camera rotation')
    if 'camera_tracking_valid' in raw:
        valid = np.asarray(raw['camera_tracking_valid'], dtype=bool)
        if valid.shape != (n,) or not valid.all():
            raise ValueError('Unreliable camera intervals: split/re-estimate camera before C-RC1')
    xyxy = np.asarray(boxes['boxes'])
    kp = np.asarray(points['keypoints'])
    if xyxy.shape != (n, 4) or kp.shape != (n, 17, 3):
        raise ValueError('No implicit truncation: align raw/video/bbox/COCO17 rows')
    if not np.isfinite(xyxy).all() or np.any(xyxy[:, 2:] <= xyxy[:, :2]):
        raise ValueError('Invalid full-image xyxy boxes')
    # ViTPose heatmap scores are NOT calibrated probabilities; the reviewed
    # detections legitimately reach ~1.04. Preserve scores, do not clamp them.
    if not np.isfinite(kp).all() or np.any(kp[:, :, 2] < 0):
        raise ValueError('Invalid pixel keypoints/scores; encode missing points with score zero')
    for name, data in [('boxes', boxes), ('keypoints', points)]:
        if 'frame' in data and not np.array_equal(data['frame'], ids):
            raise ValueError(name + '/raw frame identity mismatch')
        if 'fps' in data and not np.isclose(float(data['fps']), fps, atol=1e-3, rtol=0):
            raise ValueError(name + '/raw FPS mismatch')
        if 'valid' in data and (np.shape(data['valid']) != (n,) or not np.asarray(data['valid'], bool).all()):
            raise ValueError(name + ': split missing target intervals first')
    if 'frame' not in boxes and not np.array_equal(ids, np.arange(n)):
        raise ValueError('Nonzero original frame IDs require an explicit matching boxes.frame')
    if 'img_focal' not in boxes or 'img_center' not in boxes:
        raise ValueError('Reprojection requires the SAME img_focal/img_center used for raw inference')
    focal = np.broadcast_to(boxes['img_focal'], (n,))
    center = np.broadcast_to(boxes['img_center'], (n, 2))
    if not np.isfinite(focal).all() or np.any(focal <= 0) or not np.isfinite(center).all():
        raise ValueError('Invalid intrinsics')
    # Raw exports carry these; callers must supply the same intrinsics used by
    # the detector and preserve their file identity in the diagnostics.
    for key, actual in [('img_focal', focal), ('img_center', center)]:
        if key in raw and not np.allclose(np.broadcast_to(raw[key], actual.shape), actual, atol=1e-5, rtol=1e-6):
            raise ValueError('Intrinsics differ from raw inference: ' + key)
    return n, fps


def process_crc1(raw, spec, *, video, boxes, keypoints, confirm_single_shot=False,
                 clearance=0., cfg=CRC1Config()):
    """CPU offline solver. Return final motion, JSON report, all numerical stages.

File arguments describe exactly the same single continuous tracked shot.
No fallback is performed if any observation is missing.
"""
    start = time.monotonic()
    sources = {key: Path(value).resolve() for key, value in
               [('video', video), ('boxes', boxes), ('keypoints', keypoints)]}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    box_data, point_data = load_observations(sources['boxes'], sources['keypoints'])
    n, fps = validate_inputs(raw, spec, box_data, point_data,
                            confirm_single_shot=confirm_single_shot, clearance=clearance)
    kp = point_data['keypoints']
    # Decode/check the full video before costly optimization. Pixel thresholds
    # remain the reviewed ones; never resize the video without remapping K/boxes/kp.
    print('C-RC1 1/4: background motion and independent ankle evidence', flush=True)
    background = estimate_background(sources['video'], box_data['boxes'])
    if not np.isclose(float(background['fps']), fps, atol=1e-3, rtol=0):
        raise ValueError('Video/raw FPS mismatch')
    evidence = evidence_from_pairs(kp, box_data['boxes'], background['pair_H'], background['pair_valid'], fps)
    print('C-RC1 2/4: fixed-pose translation reprojection', flush=True)
    fitted, fit_info = fit_translation(raw, kp, box_data, accel_xy=3., accel_z=15.)
    print('C-RC1 3/4: sole-height alignment and approved balanced-v2 filter', flush=True)
    motion = dict(root_trans=np.asarray(raw['root_pos_world_zup'], np.float32).copy(),
                  root_rot_quat=np.asarray(raw['root_quat_world_zup_wxyz'], np.float32).copy(),
                  dof=np.asarray(raw['dof'], np.float32).copy(), fps=np.asarray(fps))
    human = predicted_human_sole_heights(raw['pred_pose'], raw['pred_shape'], raw['pred_trans'],
                                       raw['camera_to_world_zup_R'], raw['camera_to_world_zup_T'])
    geometry = RobotFootGeometry(spec, 'cpu')
    height, _ = geometry.evaluate(motion)
    anchored, zshift = align_root_height(motion, human, height, raw['static_prob6'][:, [1, 3]], clearance=clearance)
    filtered = balanced_motion_filter(anchored, spec, fps, so3_max_nfev=40)
    print('C-RC1 4/4: frozen image support, joint XY solve and bounded CCD', flush=True)
    # Exact reviewed order. Linearity applies fitted-root delta through the same
    # v2 smoother once; it does not refilter posture or double-filter root.
    final, weights, stage_motions, delta = refine_from_prefilter(raw, filtered, fitted, evidence, spec, fps, cfg)
    final.update(frame=raw['frame'], robot_name=np.asarray(spec.name), joint_names=np.asarray(spec.model_dof_order),
        dof_xml=model_dof_to_xml_dof(torch.tensor(final['dof']), spec).numpy(),
        coordinate_frame=np.asarray('world_zup'), quaternion_convention=np.asarray('wxyz'),
        human_sole_height_world_zup=human + delta[:, 2, None], static_prob6=raw['static_prob6'],
        static_prob6_used=weights, static_constraint_weight6=weights,
        static_constraint_policy=np.asarray(POLICY), static_head_trained=np.asarray(True),
        postprocess_applied=np.asarray(True),
        postprocess_method=np.asarray(METHOD), postprocess_mode=np.asarray('c_rc1'),
        reprojection_enabled=np.asarray(True), sole_alignment_clearance_m=np.asarray(clearance),
        gate_measurement_stage=np.asarray('Independent image evidence before support optimization; no corrected-foot self-gating'))
    for key in ('human_checkpoint_sha256', 'robot_checkpoint_sha256', 'static_checkpoint_sha256',
                'architecture', 'model_context_version', 'root_basis', 'static_version'):
        if key in raw:
            final[key] = raw[key]
    stage_motions = dict(raw=motion, foot_height_aligned=anchored, **stage_motions, final=final)
    stages = {name + '__' + key: data[key] for name, data in stage_motions.items()
              for key in ('root_trans', 'root_rot_quat', 'dof')}
    stages.update(height_shift=zshift, fitted_world_root=fitted, root_reprojection_delta=delta,
                  support_weights6=weights)
    stages.update({'crc1_background__' + key: value for key, value in background.items()})
    stages.update({'crc1_image__' + key: value for key, value in evidence.items()})
    sources['robot_asset'] = Path(spec.xml_path).resolve()
    report = dict(method=str(final['postprocess_method']), config=asdict(cfg), reprojection=fit_info, frames=n, fps=fps,
        reprojection_enabled=True,postprocess_mode='c_rc1',
        so3_max_nfev=40,root_xy_hard_cap_m=None,
        clearance_m=clearance, postprocess_device='cpu', offline=True, geometry=geometry.description,
        support_policy=POLICY, filter_policy='balanced_v2',
        clips=[dict(first_frame=int(raw['frame'][0]), last_frame=int(raw['frame'][-1]), frames=n)],
        support_foot_intervals=int((weights[:, [1, 3]] > .8).sum()),
        background_valid_fraction=float(background['pair_valid'].mean()) if n > 1 else 0.,
        sources={key: str(value) for key, value in sources.items()},
        sha256={key: sha256_file(path) for key, path in sources.items()},
        code_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in
                     ('crc1.py', 'crc1_postprocess.py', 'crc1_reprojection.py', 'crc1_image_evidence.py', 'motion_filter_v2.py')},
        elapsed_seconds=time.monotonic() - start,
        scope='Offline image-supported C-RC1 reprojection, static XY correction, and bounded IK. No GT/teacher/action rules. 2D evidence is not proof of 3D contact.',
        source_candidate='image_support_C_RC1_20261017',
        observation_identity='Matching frame metadata checked when present; plain keypoint NPY is caller-aligned.')
    return final, report, stages
