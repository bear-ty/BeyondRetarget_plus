#!/usr/bin/env python3
"""Generate world-coordinate SMPL joints from trusted dataset annotations."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


DEFAULT_SMPL_MODEL = Path(__file__).resolve().parents[1] / "data/smpl/SMPL_NEUTRAL.pkl"


def load_neutral_smpl(model_path: Path):
    """Load official NumPy model archives or explicitly supplied pickle models."""
    import smplx

    model_path = Path(model_path)
    if model_path.suffix.lower() == ".npz":
        with np.load(model_path, allow_pickle=False) as archive:
            required = {"v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{model_path}: missing SMPL model arrays {sorted(missing)}")
            data = {key: archive[key] for key in required}
        if data["v_template"].shape != (6890, 3) or data["J_regressor"].shape != (24, 6890):
            raise ValueError(f"{model_path}: expected a SMPL model with 6890 vertices and 24 joints")
        return smplx.SMPL(str(model_path), data_struct=SimpleNamespace(**data), gender="neutral")
    if model_path.suffix.lower() == ".pkl":
        return smplx.SMPL(str(model_path), gender="neutral")
    raise ValueError(f"Unsupported SMPL model format: {model_path.suffix}; use .npz or .pkl")


def load_parameters(path: Path) -> dict[str, np.ndarray]:
    """Validate axis-angle SMPL annotations without changing coordinates or shape."""
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dictionary")
    missing = {"betas", "body_pose", "global_orient", "transl"} - data.keys()
    if missing:
        raise ValueError(f"{path}: missing SMPL parameters {sorted(missing)}")
    params = {key: np.asarray(data[key], dtype=np.float32) for key in
              ("betas", "body_pose", "global_orient", "transl")}
    transl = params["transl"]
    if transl.ndim != 2 or transl.shape[1] != 3 or len(transl) == 0:
        raise ValueError("transl must have shape [T, 3] with T > 0")
    frames = len(transl)
    for key, shapes in {
        "body_pose": ((frames, 69), (frames, 23, 3)),
        "global_orient": ((frames, 3), (frames, 1, 3)),
    }.items():
        if params[key].shape not in shapes:
            raise ValueError(f"{key} must have shape {shapes}, got {params[key].shape}; use SMPL axis-angle parameters, not SMPL-X")
        params[key] = params[key].reshape(frames, -1)
    betas = params["betas"]
    if betas.ndim == 1:
        betas = betas[None]
    if betas.ndim != 2 or betas.shape[0] not in (1, frames) or betas.shape[1] < 10:
        raise ValueError("betas must have shape [K], [1, K], or [T, K], with K >= 10")
    params["betas"] = np.broadcast_to(betas[:, :10], (frames, 10))
    if not all(np.isfinite(value).all() for value in params.values()):
        raise ValueError("SMPL parameters contain NaN or Inf")
    return params


class SMPLKeypointGenerator:
    """Reuse one neutral model across sequences and bound memory by batch size."""

    def __init__(self, model_path: Path = DEFAULT_SMPL_MODEL, batch_size: int = 64, device: str = "cpu"):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_path = Path(model_path).expanduser().resolve()
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = None

    def require_model_file(self) -> None:
        """Check the required download before starting dataset preparation."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SMPL model missing: {self.model_path}. Register at https://smpl.is.tue.mpg.de/, "
                "accept the model license, and download the gender-neutral SMPL model. "
                "Place SMPL_NEUTRAL.pkl at the default path or pass --smpl_model."
            )

    def generate(self, sequence_dir: Path, overwrite: bool = False) -> Path:
        sequence_dir = Path(sequence_dir)
        output = sequence_dir / "keypoints.npy"
        params = load_parameters(sequence_dir / "smpl.npy")
        frames = len(params["transl"])
        if output.exists() and not overwrite:
            joints = np.load(output, allow_pickle=False)
            if joints.shape != (frames, 24, 3) or not np.isfinite(joints).all():
                raise ValueError(f"{output}: expected finite [{frames}, 24, 3]; use --overwrite to regenerate")
            return output
        if self.model is None:
            self.require_model_file()
            try:
                self.model = load_neutral_smpl(self.model_path).to(self.device).eval()
            except ModuleNotFoundError as exc:
                if exc.name == "chumpy":
                    raise RuntimeError(
                        "This SMPL pickle contains legacy Chumpy objects. Convert it using the official "
                        "smplx tools/clean_ch.py workflow: https://github.com/vchoutas/smplx#model-loading"
                    ) from exc
                raise
        joints = np.empty((frames, 24, 3), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, frames, self.batch_size):
                stop = min(start + self.batch_size, frames)
                batch = {key: torch.tensor(value[start:stop], device=self.device) for key, value in params.items()}
                result = self.model(**batch, return_verts=False)
                joints[start:stop] = result.joints[:, :24].cpu().numpy()
        if not np.isfinite(joints).all():
            raise ValueError(f"{sequence_dir}: SMPL produced NaN or Inf")
        # Replace the destination only after the complete array has been written.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=sequence_dir, suffix=".npy", delete=False) as handle:
                temp_path = Path(handle.name)
                np.save(handle, joints, allow_pickle=False)
            os.replace(temp_path, output)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--smpl_model", type=Path, default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.dataset_root.expanduser().resolve().rglob("smpl.npy"))
    if not files:
        parser.error(f"No smpl.npy found under {args.dataset_root}")
    generator = SMPLKeypointGenerator(args.smpl_model, args.batch_size, args.device)
    generator.require_model_file()
    failures = 0
    for path in files:
        try:
            print(f"ready: {generator.generate(path.parent, overwrite=args.overwrite)}")
        except Exception as exc:
            failures += 1
            print(f"failed: {path}: {exc}")
    print(f"ready={len(files) - failures} failed={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
