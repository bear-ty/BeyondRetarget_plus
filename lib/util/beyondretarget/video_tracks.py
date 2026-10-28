"""Validation for the single-person box-track NPZ shared by video utilities."""
from pathlib import Path

import numpy as np


def load_box_track(path, require_complete=False):
    """Load boxes/valid/frame without pickle and enforce temporal alignment."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Box track not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = {"boxes", "valid", "frame"}.difference(payload.files)
        if missing:
            raise ValueError(f"Box track {path} lacks fields: {sorted(missing)}")
        track = {name: payload[name] for name in ("boxes", "valid", "frame")}
    boxes = np.asarray(track["boxes"], dtype=np.float32)
    valid = np.asarray(track["valid"], dtype=bool)
    frame = np.asarray(track["frame"], dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) < 1:
        raise ValueError(f"boxes must be non-empty [T,4], got {boxes.shape}")
    if valid.shape != (len(boxes),) or frame.shape != (len(boxes),):
        raise ValueError("valid and frame must both have shape [T]")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes contain NaN or Inf")
    if not np.array_equal(frame, np.arange(len(boxes), dtype=np.int64)):
        raise ValueError("frame must be exactly 0..T-1; split the video/track at gaps")
    observed = valid & ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    if not np.array_equal(observed, valid):
        raise ValueError("Every valid observation must have positive box width and height")
    if require_complete and not valid.all():
        raise ValueError(
            f"Track has {int((~valid).sum())} unobserved frames; split at gaps before complete C-RC1 inference"
        )
    return {"boxes": boxes, "valid": valid, "frame": frame}
