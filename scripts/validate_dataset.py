#!/usr/bin/env python3
"""Validate the prepared RGB2Robo training or evaluation dataset layout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.contact_labels import normalize_contact_lr


REQUIRED_FILES = ("keypoints.npy", "smpl.npy")


def discover_sequences(root: Path) -> list[Path]:
    """Find prepared, raw, and empty sequences without descending into frame folders."""
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    markers = set(REQUIRED_FILES) | {"vit_features.pt", "contact.npy", "bbox.npy"}
    sequences = []
    visited = set()

    def raise_walk_error(error):
        raise error

    for directory, children, files in os.walk(root, followlinks=True, onerror=raise_walk_error):
        path = Path(directory)
        resolved = path.resolve()
        if resolved in visited:
            children[:] = []
            continue
        visited.add(resolved)
        children[:] = sorted(name for name in children if not name.startswith("."))
        has_video = any(Path(name).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} for name in files)
        if markers.intersection(files) or "color" in children or has_video or not children:
            sequences.append(path)
            children[:] = []
    return sorted(sequences)


def validate_sequence(
    sequence_dir: Path, require_contact: bool = False, require_vit_features: bool = False
) -> tuple[list[str], np.ndarray | None]:
    errors = []
    arrays = {}
    lengths = {}
    unavailable = object()

    def load(name, loader, required=True):
        path = sequence_dir / name
        if not path.is_file():
            if required:
                errors.append(f"missing {name}")
            return unavailable
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"cannot load {name}: {type(exc).__name__}: {exc}")
            return unavailable

    def check_array(name, value, shapes):
        try:
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise ValueError(f"must contain real numeric values, got {array.dtype}")
            if array.ndim < 1 or array.shape[1:] not in shapes:
                raise ValueError(f"invalid shape {array.shape}; expected [T, ...] with trailing shape in {shapes}")
            if not np.isfinite(array).all():
                errors.append(f"{name} contains NaN or Inf")
            lengths[name] = len(array)
            arrays[name] = array
        except (TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")

    features = load(
        "vit_features.pt",
        lambda path: torch.load(path, map_location="cpu", weights_only=True),
        required=require_vit_features,
    )
    if features is not unavailable:
        if (
            not isinstance(features, torch.Tensor) or features.layout != torch.strided
            or features.is_complex() or features.is_quantized or features.device.type != "cpu"
        ):
            errors.append("vit_features.pt must contain a dense tensor [T, 1024]")
        else:
            features = features.detach()
            if features.dtype == torch.bfloat16:
                features = features.float()
            check_array("vit_features.pt", features, {(1024,)})
    keypoints = load("keypoints.npy", np.load)
    if keypoints is not unavailable:
        check_array("keypoints.npy", keypoints, {(24, 3)})
    smpl = load("smpl.npy", lambda path: np.load(path, allow_pickle=True).item())
    if smpl is not unavailable:
        if not isinstance(smpl, dict):
            errors.append("smpl.npy must contain a parameter dictionary")
        else:
            for key, shapes in (("transl", {(3,)}), ("global_orient", {(3,), (1, 3)})):
                if key not in smpl:
                    errors.append(f"smpl.npy missing {key}")
                else:
                    check_array(f"smpl.npy {key}", smpl[key], shapes)
            if "body_pose" in smpl:
                check_array("smpl.npy body_pose", smpl["body_pose"], {(69,), (23, 3)})
            if "betas" in smpl:
                try:
                    betas = np.asarray(smpl["betas"])
                    if betas.ndim == 1:
                        betas = betas[None]
                    if betas.ndim != 2 or betas.shape[1] < 10 or betas.shape[0] < 1:
                        raise ValueError("must be [K], [1, K], or [T, K], with K >= 10")
                    if betas.dtype.kind not in "biuf" or not np.isfinite(betas).all():
                        raise ValueError("must contain finite real numeric values")
                    if len(betas) != 1:
                        lengths["smpl.npy betas"] = len(betas)
                except (TypeError, ValueError) as exc:
                    errors.append(f"smpl.npy betas: {exc}")
    if not (sequence_dir / "keypoints.npy").is_file() and (sequence_dir / "smpl.npy").is_file():
        errors.append("generate missing joints with scripts/generate_smpl_keypoints.py --dataset_root PATH (see README)")
    bbox = load("bbox.npy", np.load, required=False)
    if bbox is not unavailable:
        check_array("bbox.npy", bbox, {(8,)})
        if "bbox.npy" in arrays and np.any(arrays["bbox.npy"][:, 3:5] <= arrays["bbox.npy"][:, 1:3]):
            errors.append("bbox.npy must have positive box widths and heights")
    contact_values = load("contact.npy", np.load, required=require_contact)
    contact = None
    if contact_values is not unavailable:
        try:
            contact = normalize_contact_lr(contact_values)
            lengths["contact.npy"] = len(contact)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid contact.npy: {exc}")
    if lengths and (min(lengths.values()) <= 0 or len(set(lengths.values())) != 1):
        errors.append(f"frame-count mismatch or empty sequence: {lengths}")
    return errors, contact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-contact", action="store_true", help="Fail if any sequence lacks contact.npy")
    parser.add_argument(
        "--require-vit-features",
        action="store_true",
        help="Also require legacy RGB2Robo vit_features.pt; not used by the Plus frozen-feature cache",
    )
    args = parser.parse_args()
    try:
        sequence_dirs = discover_sequences(args.root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not sequence_dirs:
        parser.error(f"No sequences found under {args.root}")
    failures = {}
    positive_counts = np.zeros(2, dtype=np.int64)
    contact_frames = 0
    for sequence_dir in sequence_dirs:
        errors, contact = validate_sequence(
            sequence_dir,
            require_contact=args.require_contact,
            require_vit_features=args.require_vit_features,
        )
        if errors:
            failures[str(sequence_dir)] = errors
        elif contact is not None:
            positive_counts += (contact > 0.5).sum(axis=0)
            contact_frames += len(contact)
    print(f"validated={len(sequence_dirs)} invalid={len(failures)}")
    if contact_frames:
        rates = positive_counts.astype(np.float64) / float(contact_frames)
        print(
            f"contact_frames={contact_frames} "
            f"left_positive={positive_counts[0]} ({rates[0]:.6f}) "
            f"right_positive={positive_counts[1]} ({rates[1]:.6f})"
        )
    if failures:
        for path, errors in failures.items():
            print(f"{path}: {'; '.join(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
