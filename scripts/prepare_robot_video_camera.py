"""TRAM official SAM/masked DROID/ZoeDepth/SPEC using an existing target bbox track.

The detector/identity track is reused, not the old model's pose or trajectory.
All saved cameras have metric scale and the official gravity alignment.
"""
import argparse, gc, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'thirdparty/DROID-SLAM'))
sys.path.insert(0, str(ROOT / 'thirdparty/DROID-SLAM/thirdparty/lietorch'))
# ZoeDepth/MiDaS requires legacy timm; isolate it to this camera subprocess.
# Robot inference/training keeps its original timm and checkpoint semantics.
if os.environ.get('TRAM_CAMERA_DEPS'):
    sys.path.insert(0, os.environ['TRAM_CAMERA_DEPS'])

import cv2
import numpy as np
import torch

from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.video_tracks import load_box_track


CACHE_IDENTITY_VERSION = 1
CACHE_ARTIFACTS = (
    'images', 'human_masks.npy', 'intrinsics.npz', 'camera.npy',
    'camera_provenance.json', 'boxes.npz',
)


def make_cache_identity(video, bbox_source, camera_mode):
    return {
        'version': CACHE_IDENTITY_VERSION,
        'input_video_sha256': sha256_file(video),
        'bbox_sha256': sha256_file(bbox_source),
        'requested_mode': camera_mode,
    }


def prepare_output_cache(out, identity):
    """Bind every reusable artifact to one video, track, and requested mode."""
    identity_path = out / 'cache_identity.json'
    if identity_path.exists():
        try:
            stored = json.loads(identity_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f'Camera cache identity is unreadable: {identity_path}; use a new output directory.'
            ) from exc
        if stored != identity:
            raise ValueError(
                'Camera cache input video, track, or camera mode does not match; '
                'use a new output directory.'
            )
        return
    if any((out / name).exists() for name in CACHE_ARTIFACTS):
        raise ValueError(
            'Legacy camera cache has no verifiable input/track/mode identity; '
            'use a new output directory.'
        )
    temporary = out / '.cache_identity.json.tmp'
    temporary.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    temporary.replace(identity_path)


def extract_frames(video, frames, expected_count):
    frames.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot read input video: {video}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Input video has invalid FPS: {fps}')
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            destination = frames / f'{count:06d}.jpg'
            if not destination.exists() and not cv2.imwrite(str(destination), image):
                raise IOError(f'Frame extraction failed: {destination}')
            count += 1
    finally:
        cap.release()
    if count != expected_count or count < 16:
        raise ValueError(
            f'Video/track frame mismatch or too few frames: video={count}, track={expected_count}; '
            'at least 16 aligned frames are required'
        )
    expected_names = [f'{index:06d}.jpg' for index in range(count)]
    actual = sorted(frames.glob('*.jpg'), key=lambda path: path.name)
    if [path.name for path in actual] != expected_names:
        raise ValueError(f'Frame cache is incomplete or contains unexpected images: {frames}')
    frame_shape = None
    for path in actual:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cached frame cannot be decoded: {path}')
        if frame_shape is None:
            frame_shape = image.shape[:2]
        elif image.shape[:2] != frame_shape:
            raise ValueError(f'Cached frame dimensions differ: {path}')
    return [str(path) for path in actual], fps, frame_shape


def validate_masks(masks, frame_count, frame_shape):
    masks = np.asarray(masks)
    expected = (frame_count, *frame_shape)
    if masks.shape != expected:
        raise ValueError(f'human_masks.npy must have shape {expected}, got {masks.shape}')
    if masks.dtype.kind not in 'buif' or not np.isfinite(masks).all():
        raise ValueError('human_masks.npy must contain finite numeric or boolean values')
    if not np.logical_or(masks == 0, masks == 1).all():
        raise ValueError('human_masks.npy must contain binary masks')
    return masks


def load_masks(path, frame_count, frame_shape):
    try:
        return validate_masks(np.load(path, allow_pickle=False), frame_count, frame_shape)
    except Exception as exc:
        raise ValueError(f'Invalid human mask cache {path}: {exc}; use a new output directory.') from exc


def validate_intrinsics(calib, detected_static, stored_mode, requested_mode):
    calib = np.asarray(calib)
    if calib.shape != (4,) or not np.isfinite(calib).all() or np.any(calib[:2] <= 0):
        raise ValueError('calib must be finite [fx,fy,cx,cy] with positive focal lengths')
    static_array = np.asarray(detected_static)
    if static_array.shape != () or static_array.dtype.kind != 'b':
        raise ValueError('detected_static must be a scalar boolean')
    mode_array = np.asarray(stored_mode)
    if mode_array.shape != ():
        raise ValueError('Intrinsics mode must be a scalar string')
    if str(mode_array.item()) != requested_mode:
        raise ValueError(
            f'Intrinsics mode {mode_array.item()!r} does not match requested mode {requested_mode!r}'
        )
    return calib, bool(static_array.item())


def load_intrinsics(path, requested_mode):
    try:
        with np.load(path, allow_pickle=False) as cache:
            return validate_intrinsics(
                cache['calib'], cache['detected_static'], cache['mode'], requested_mode,
            )
    except Exception as exc:
        raise ValueError(f'Invalid intrinsics cache {path}: {exc}; use a new output directory.') from exc


def validate_camera(camera, frame_count, calib=None):
    if not isinstance(camera, dict):
        raise ValueError('camera.npy must contain a dictionary')
    required = {
        'pred_cam_R', 'pred_cam_T', 'world_cam_R', 'world_cam_T',
        'img_focal', 'img_center', 'spec_focal',
    }
    missing = required.difference(camera)
    if missing:
        raise ValueError(f'camera.npy lacks fields: {sorted(missing)}')
    normalized = dict(camera)
    for key in ('pred_cam_R', 'world_cam_R'):
        rotation = np.asarray(camera[key])
        if rotation.shape != (frame_count, 3, 3):
            raise ValueError(f'{key} must have shape {(frame_count, 3, 3)}, got {rotation.shape}')
        if not np.isfinite(rotation).all():
            raise ValueError(f'{key} contains NaN or Inf')
        check = rotation.astype(np.float64)
        if not np.allclose(
            check @ check.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-3,
        ) or not np.allclose(np.linalg.det(check), 1.0, rtol=0, atol=1e-3):
            raise ValueError(f'{key} is not a proper rotation')
        normalized[key] = rotation
    for key in ('pred_cam_T', 'world_cam_T'):
        translation = np.asarray(camera[key])
        if translation.shape != (frame_count, 3) or not np.isfinite(translation).all():
            raise ValueError(f'{key} must be finite with shape {(frame_count, 3)}')
        normalized[key] = translation
    for key in ('img_focal', 'spec_focal'):
        focal = np.asarray(camera[key])
        if focal.shape != () or not np.isfinite(focal).all() or float(focal) <= 0:
            raise ValueError(f'{key} must be a positive finite scalar')
        normalized[key] = focal
    center = np.asarray(camera['img_center'])
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('img_center must be a finite [cx,cy] array')
    normalized['img_center'] = center
    if calib is not None:
        if not np.isclose(float(normalized['img_focal']), float(calib[0]), rtol=1e-5, atol=1e-5):
            raise ValueError('camera img_focal does not match cached intrinsics')
        if not np.allclose(center, calib[2:], rtol=1e-5, atol=1e-5):
            raise ValueError('camera img_center does not match cached intrinsics')
    return normalized


def load_camera(path, frame_count, calib):
    try:
        payload = np.load(path, allow_pickle=True)
        if not isinstance(payload, np.ndarray) or payload.shape != ():
            raise ValueError('camera.npy must contain one dictionary object')
        return validate_camera(payload.item(), frame_count, calib)
    except Exception as exc:
        raise ValueError(f'Invalid camera cache {path}: {exc}; use a new output directory.') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--boxes', help='Single-person NPZ from scripts/extract_video_boxes.py')
    source.add_argument('--bbox-pt', help='Legacy PT dictionary containing bbx_xyxy')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-mode', choices=['moving', 'static', 'auto'], default='moving')
    args = parser.parse_args()
    bbox_source = args.boxes or args.bbox_pt
    if args.boxes:
        track = load_box_track(args.boxes, require_complete=True)
        boxes = track['boxes']
        track_valid = track['valid']
        bbox_payload = {'bbx_xyxy': torch.from_numpy(boxes)}
    else:
        bbox_payload = torch.load(args.bbox_pt, map_location='cpu', weights_only=False)
        boxes = bbox_payload['bbx_xyxy'].detach().cpu().numpy().astype(np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError('bbx_xyxy must be a finite [T,4] tensor')
        track_valid = np.ones(len(boxes), dtype=bool)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity = make_cache_identity(args.video, bbox_source, args.camera_mode)
    prepare_output_cache(out, identity)
    torch.set_num_threads(4)
    imgfiles, fps, frame_shape = extract_frames(args.video, out / 'images', len(boxes))
    frame_count = len(imgfiles)

    mask_path = out / 'human_masks.npy'
    if mask_path.exists():
        masks_np = load_masks(mask_path, frame_count, frame_shape)
    else:
        from segment_anything import SamPredictor, sam_model_registry
        print('SAM foreground masks from verified bbox track', flush=True)
        model = sam_model_registry['vit_h'](
            checkpoint=str(ROOT / 'data/pretrain/sam_vit_h_4b8939.pth')
        ).cuda().eval()
        predictor = SamPredictor(model)
        masks = []
        with torch.inference_mode():
            for index, file in enumerate(imgfiles):
                predictor.set_image(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
                mask_boxes = bbox_payload.get('all_boxes', [boxes[j:j + 1] for j in range(frame_count)])[index]
                mask_boxes = np.concatenate([
                    np.asarray(mask_boxes).reshape(-1, 4), boxes[index:index + 1],
                ])
                mask_list = []
                for box in mask_boxes:
                    mask, _, _ = predictor.predict(box=box, multimask_output=False)
                    mask_list.append(mask[0])
                masks.append(np.logical_or.reduce(mask_list))
                if index % 30 == 0:
                    print(f'SAM {index + 1}/{frame_count}', flush=True)
        np.save(mask_path, np.stack(masks))
        masks_np = load_masks(mask_path, frame_count, frame_shape)
        del predictor, model, masks
        gc.collect()
        torch.cuda.empty_cache()

    # Official depth-scale estimation uses cv2.resize (does not accept bool).
    # Keep the saved binary masks, pass 0/1 float masks to the official frontend.
    masks = torch.from_numpy(masks_np).float()
    intrinsics_path = out / 'intrinsics.npz'
    camera_path = out / 'camera.npy'
    if camera_path.exists() and not intrinsics_path.exists():
        raise ValueError(
            'Camera cache has camera.npy but no verifiable intrinsics.npz; use a new output directory.'
        )
    calib = detected_static = None
    if intrinsics_path.exists():
        calib, detected_static = load_intrinsics(intrinsics_path, args.camera_mode)

    if camera_path.exists():
        camera = load_camera(camera_path, frame_count, calib)
    else:
        from lib.camera import calibrate_intrinsics, run_metric_slam, align_cam_to_world
        print('Official TRAM intrinsic search + metric masked SLAM', flush=True)
        if calib is None:
            calib, detected_static = calibrate_intrinsics(
                str(out / 'images'), masks, is_static=args.camera_mode == 'static',
            )
            calib, detected_static = validate_intrinsics(
                calib, np.asarray(detected_static), np.asarray(args.camera_mode), args.camera_mode,
            )
            np.savez(
                intrinsics_path, calib=calib, detected_static=detected_static, mode=args.camera_mode,
            )
        is_static = detected_static if args.camera_mode == 'auto' else args.camera_mode == 'static'
        rotation, translation = run_metric_slam(
            str(out / 'images'), masks=masks, calib=calib, is_static=is_static,
        )
        world_rotation, world_translation, spec_focal = align_cam_to_world(
            imgfiles[0], rotation, translation,
        )
        camera = validate_camera({
            'pred_cam_R': rotation.numpy(), 'pred_cam_T': translation.numpy(),
            'world_cam_R': world_rotation.numpy(), 'world_cam_T': world_translation.numpy(),
            'img_focal': np.asarray(calib[0]), 'img_center': np.asarray(calib[2:]),
            'spec_focal': spec_focal,
        }, frame_count, calib)
        np.save(camera_path, camera)
        (out / 'camera_provenance.json').write_text(json.dumps(dict(
            requested_mode=args.camera_mode, auto_detected_static=bool(detected_static),
            used_static=bool(is_static), method='official_TRAM_masked_DROID_ZoeDepth_SPEC',
            segmentation='official_SAM_vit_h_target_bbox',
            bbox_source=str(Path(bbox_source).resolve()), bbox_sha256=identity['bbox_sha256'],
            input_video_sha256=identity['input_video_sha256'], frames=frame_count, fps=fps,
            checkpoint_sha256={name: sha256_file(ROOT / 'data/pretrain' / name) for name in
                               ['droid.pth', 'camcalib_sa_biased_l2.ckpt', 'sam_vit_h_4b8939.pth']},
        ), indent=2))

    np.savez_compressed(
        out / 'boxes.npz', boxes=boxes, valid=track_valid, frame=np.arange(frame_count),
        img_focal=camera['img_focal'], img_center=camera['img_center'],
    )
    print(json.dumps(dict(
        complete=True, frames=frame_count, fps=fps, output=str(out),
    ), indent=2), flush=True)


if __name__ == '__main__':
    main()
