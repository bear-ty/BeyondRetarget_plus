"""Extract a single-person bounding-box track with Ultralytics YOLO."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--weights", default="assets/yolo/yolov8x.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(video)
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))
    results = model.track(
        source=str(video),
        classes=[0],
        conf=args.confidence,
        iou=args.iou,
        device=args.device,
        persist=True,
        stream=True,
        verbose=False,
    )
    boxes_by_frame = []
    tracks_by_frame = []
    for result in results:
        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        confidence = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        track_ids = result.boxes.id.cpu().numpy().astype(np.int64) if result.boxes is not None and result.boxes.id is not None else np.full(len(xyxy), -1, dtype=np.int64)
        boxes_by_frame.append(xyxy.astype(np.float32))
        tracks_by_frame.append((track_ids, confidence.astype(np.float32)))
    if not boxes_by_frame:
        raise RuntimeError("The video contains no decoded frames")

    counts = {}
    for track_ids, _ in tracks_by_frame:
        for track_id in track_ids:
            if track_id >= 0:
                counts[int(track_id)] = counts.get(int(track_id), 0) + 1
    selected_id = max(counts, key=counts.get) if counts else -1
    boxes = np.full((len(boxes_by_frame), 4), np.nan, dtype=np.float32)
    confidence = np.zeros(len(boxes_by_frame), dtype=np.float32)
    valid = np.zeros(len(boxes_by_frame), dtype=bool)
    for index, (frame_boxes, (track_ids, frame_conf)) in enumerate(zip(boxes_by_frame, tracks_by_frame)):
        candidates = np.flatnonzero(track_ids == selected_id) if selected_id >= 0 else np.arange(len(frame_boxes))
        if len(candidates):
            candidate = candidates[np.argmax(frame_conf[candidates])]
            boxes[index] = frame_boxes[candidate]
            confidence[index] = frame_conf[candidate]
            valid[index] = True
    if not valid.any():
        raise RuntimeError("No person track was found")

    # Fill short detector gaps for downstream continuous-window validation.
    frame_ids = np.arange(len(boxes), dtype=np.int64)
    valid_indices = np.flatnonzero(valid)
    for index in np.flatnonzero(~valid):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        boxes[index] = boxes[nearest]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, boxes=boxes, valid=valid, frame=frame_ids, confidence=confidence)
    print(f"saved {output} ({len(boxes)} frames, {valid.sum()} detector observations, track_id={selected_id})")


if __name__ == "__main__":
    main()
