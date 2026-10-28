"""Generate aligned image-space COCO17 keypoints with an Ultralytics pose model."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.util.beyondretarget.video_tracks import load_box_track


def select_pose_box(target, candidates):
    """Return the candidate with the highest IoU and that IoU."""
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 4)
    if not len(candidates):
        return -1, 0.0
    top_left = np.maximum(candidates[:, :2], target[:2])
    bottom_right = np.minimum(candidates[:, 2:], target[2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    target_area = max(float(target[2] - target[0]), 0.0) * max(float(target[3] - target[1]), 0.0)
    candidate_areas = np.maximum(candidates[:, 2] - candidates[:, 0], 0) * np.maximum(
        candidates[:, 3] - candidates[:, 1], 0
    )
    union = target_area + candidate_areas - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    index = int(np.argmax(iou))
    return index, float(iou[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--boxes', required=True, help='Raw single-person NPZ from extract_video_boxes.py')
    parser.add_argument('--weights', default='assets/yolo/yolov8x-pose.pt')
    parser.add_argument('--output', required=True, help='Destination COCO17 .npz')
    parser.add_argument('--device', default=None)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7, help='Pose-model NMS IoU')
    parser.add_argument('--min-track-iou', type=float, default=0.1, help='Minimum IoU with the target track box')
    args = parser.parse_args()

    video = Path(args.video)
    weights = Path(args.weights)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not weights.is_file():
        raise FileNotFoundError(
            f'Pose weights not found: {weights}; see README for the official Ultralytics download link'
        )
    track = load_box_track(args.boxes, require_complete=True)
    if len(track['boxes']) < 16:
        raise ValueError('Complete Plus inference requires at least 16 frames')
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'Could not read a valid video FPS from {video}')

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=str(video), classes=[0], conf=args.confidence, iou=args.iou,
        device=args.device, stream=True, verbose=False,
    )
    count = len(track['boxes'])
    keypoints = np.zeros((count, 17, 3), dtype=np.float32)
    observed = np.zeros(count, dtype=bool)
    decoded = 0
    for index, result in enumerate(results):
        if index >= count:
            raise RuntimeError(f'Pose model decoded more than the {count} track frames')
        decoded += 1
        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        candidate, iou = select_pose_box(track['boxes'][index], boxes)
        if candidate < 0 or iou < args.min_track_iou or result.keypoints is None:
            continue
        xy = result.keypoints.xy.detach().cpu().numpy()
        scores = result.keypoints.conf.detach().cpu().numpy()
        if xy.shape != scores.shape + (2,) or len(xy) != len(boxes):
            raise ValueError(f'Unexpected pose output shapes at frame {index}: {xy.shape}, {scores.shape}')
        if xy.shape[1:] != (17, 2):
            raise ValueError(f'Expected COCO17 keypoints, got {xy.shape} at frame {index}')
        candidate_points = np.concatenate([xy[candidate], scores[candidate, :, None]], axis=1)
        if not np.isfinite(candidate_points).all() or (candidate_points[:, 2] < 0).any():
            raise ValueError(f'Non-finite or negative keypoint values at frame {index}')
        keypoints[index] = candidate_points
        observed[index] = True
    if decoded != count:
        raise RuntimeError(f'Pose model decoded {decoded} frames, expected {count}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, keypoints=keypoints, frame=track['frame'], fps=np.asarray(fps), valid=observed,
    )
    if not observed.all():
        raise RuntimeError(
            f'Could not associate the target person in {int((~observed).sum())} frames; '
            f'inspect {output} or split/re-extract the track before complete inference'
        )
    print(f'saved {output} ({count} frames, COCO17 xy in original-image pixels, fps={fps})')


if __name__ == '__main__':
    main()
