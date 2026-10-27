#!/usr/bin/env python3
"""Generate MotionPRO left/right forefoot contact labels from pressure and SMPL joints."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def pressure_to_contact(pressure: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Apply MotionPRO's pressure threshold and joint-height gate in carpet coordinates.

    The 160-by-120 mat spans 2 m along negative Y and 1.5 m along positive X.
    Pressure images are flipped along both axes. Each forefoot samples a
    radius-3, upper-exclusive region around its nearest cell; contact requires
    a pressure sum above 100 and joint Z below 0.1 m.
    """
    pressure = np.asarray(pressure)
    joints = np.asarray(joints)
    if pressure.ndim != 3 or pressure.shape[1:] != (160, 120) or len(pressure) == 0:
        raise ValueError(f"pressure must have shape [T, 160, 120] with T > 0, got {pressure.shape}")
    if pressure.dtype.kind not in "biuf":
        raise ValueError("pressure must contain real numeric values")
    if joints.shape != (len(pressure), 24, 3):
        raise ValueError(f"keypoints must have shape [{len(pressure)}, 24, 3], got {joints.shape}")
    if joints.dtype.kind not in "biuf" or not np.isfinite(joints).all():
        raise ValueError("keypoints must contain finite real numeric values")

    height, width = pressure.shape[1:]
    scale = 2.0 / height
    x_centers = np.arange(width) * scale + 0.5 * scale
    y_centers = -np.arange(height) * scale - 0.5 * scale
    feet = joints[:, [10, 11], :]
    # The Cartesian grid makes nearest-cell search separable along X and Y.
    x = np.abs(feet[:, :, 0, None] - x_centers).argmin(axis=-1)
    y = np.abs(feet[:, :, 1, None] - y_centers).argmin(axis=-1)
    contact = np.zeros((len(pressure), 2), dtype=np.float32)
    for frame in range(len(pressure)):
        values = pressure[frame, ::-1, ::-1]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"pressure frame {frame} contains nonfinite or negative values")
        for side in range(2):
            if feet[frame, side, 2] >= 0.1:
                continue
            cx, cy = x[frame, side], y[frame, side]
            # Clip the region so negative slices cannot wrap around the mat.
            region = values[max(0, cy - 3):min(height, cy + 3),
                            max(0, cx - 3):min(width, cx + 3)]
            contact[frame, side] = region.sum(dtype=np.float64) > 100
    return contact


def generate_contact_labels(sequence_dir: Path, overwrite: bool = False) -> Path:
    """Write two-column contacts atomically, reusing current validated outputs."""
    sequence_dir = Path(sequence_dir)
    pressure_path = sequence_dir / "pressure.npz"
    joints_path = sequence_dir / "keypoints.npy"
    output = sequence_dir / "contact.npy"
    joints = np.load(joints_path, allow_pickle=False)
    with np.load(pressure_path, allow_pickle=False) as archive:
        if "pressure" not in archive:
            raise ValueError(f"{pressure_path}: missing 'pressure' array")
        contact = pressure_to_contact(archive["pressure"], joints)
    if output.exists() and not overwrite:
        existing = np.load(output, allow_pickle=False)
        if existing.shape == contact.shape and np.array_equal(existing, contact):
            return output
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=sequence_dir, suffix=".npy", delete=False) as handle:
            temp_path = Path(handle.name)
            np.save(handle, contact, allow_pickle=False)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.dataset_root.expanduser().resolve().rglob("pressure.npz"))
    if not files:
        parser.error(f"No pressure.npz found under {args.dataset_root}")
    failures = 0
    for path in files:
        try:
            print(f"ready: {generate_contact_labels(path.parent, args.overwrite)}")
        except Exception as exc:
            failures += 1
            print(f"failed: {path.parent}: {exc}")
    print(f"ready={len(files) - failures} failed={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
