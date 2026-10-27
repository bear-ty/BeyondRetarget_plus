#!/usr/bin/env python3
"""Evaluate robot predictions with the metrics used in the BeyondRetarget paper.

The reported names and units intentionally follow the paper: ``MAMPJPE`` and
``RTE`` are millimetres, ``TDE``/``EDE`` are degrees, ``Jitter`` is expressed in
units of 10 m/s^3, ``Accel`` in mm/frame^2, ``FS`` in mm/frame, and ``Viol`` in
percent.  ``Fail65``, ``Fail100``, ``Exec. SR`` and ``Exec. MAMPJPE`` require
the paper's fixed evaluation windows or a MuJoCo/SONIC execution log; they are
reported as unavailable when those inputs are not supplied rather than being
silently approximated.

The evaluator accepts the raw camera-frame NPZ written by
``scripts/infer_beyondretarget.py`` (``root_pos``/``root_rot_6d``).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.robot.robot_kinematics import RobotKinematicsModel  # noqa: E402
from lib.robot.robot_spec import load_robot_spec  # noqa: E402
from lib.util.beyondretarget.gamp_pa_mpjpe_utils import apply_torso_spine2_proxy, pa_align_sequence, scale_shared_to_robot_shape  # noqa: E402
from lib.robot import torch_utils as robot_torch_utils  # noqa: E402


ROBOT_FOOT_JOINT_NAMES = {
    "g1": ("left_foot", "right_foot"),
    "r1": ("left_foot", "right_foot"),
}


# Keep this table next to the implementation so CSV/JSON consumers can check
# units without reverse-engineering abbreviated column names.
PAPER_METRIC_UNITS = {
    "MAMPJPE": "mm",
    "RTE": "mm",
    "TDE": "deg",
    "EDE": "deg",
    "Jitter": "10 m/s^3",
    "Accel": "mm/frame^2",
    "Fail65": "%",
    "Fail100": "%",
    "FS": "mm/frame",
    "Viol": "%",
    "Exec. SR": "%",
    "Exec. MAMPJPE": "mm",
}

PAPER_METRIC_DEFINITIONS = {
    "MAMPJPE": "Morphology-Aligned MPJPE after per-frame similarity alignment.",
    "RTE": "Root Trajectory Error after one rigid alignment of the full sequence.",
    "TDE": "Torso direction angular error relative to the SMPL-derived target.",
    "EDE": "Mean left/right end-effector direction angular error relative to target.",
    "Jitter": "Prediction jerk magnitude, scaled by 10 m/s^3.",
    "Accel": "Second-difference acceleration error against target.",
    "Fail65": "Percent of complete 200-frame windows with mean MAMPJPE > 65 mm.",
    "Fail100": "Percent of complete 200-frame windows with mean MAMPJPE > 100 mm.",
    "FS": "Horizontal support-foot displacement per frame using external support labels.",
    "Viol": "Contact frames with floating (>2 cm) or penetrating (<-3 cm) feet.",
    "Exec. SR": "Successful MuJoCo/SONIC execution clips divided by total clips.",
    "Exec. MAMPJPE": "MAMPJPE measured on executed trajectories.",
}


def finite_or_none(value):
    """Serialize unavailable numeric metrics as JSON null, never NaN."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def _shared_markers_from_body(body: torch.Tensor, body_rot: torch.Tensor, kin: RobotKinematicsModel, spec) -> torch.Tensor:
    link_indices = [kin.body_names.index(item.robot_link) for item in spec.shared_joint_map]
    shared = body[:, link_indices]
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0] if item.local_offset is None else [float(value) for value in item.local_offset]
            for item in spec.shared_joint_map
        ],
        dtype=shared.dtype,
        device=shared.device,
    )
    if torch.any(offsets != 0):
        marker_rot = body_rot[:, link_indices]
        flat_offsets = offsets.unsqueeze(0).expand(shared.shape[0], -1, -1).reshape(-1, 3)
        rotated_offsets = robot_torch_utils.quat_rotate(marker_rot.reshape(-1, 4), flat_offsets).reshape_as(shared)
        shared = shared + rotated_offsets
    return shared


def axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.clip(angle, 1e-8, None)
    half = 0.5 * angle
    quat = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    small = angle[..., 0] < 1e-8
    quat[small] = np.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype)
    return quat.astype(np.float32)


def load_motion_arrays(data: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the canonical raw prediction fields for FK evaluation."""
    if "dof" not in data:
        raise ValueError("Prediction NPZ is missing dof")
    dof = np.asarray(data["dof"], dtype=np.float32)
    if "root_pos" in data:
        root = np.asarray(data["root_pos"], dtype=np.float32)
    else:
        raise ValueError("Prediction NPZ needs root_pos")
    if "root_rot_6d" in data:
        rows = torch.as_tensor(np.asarray(data["root_rot_6d"], dtype=np.float32))
        # The inference exporter writes PyTorch3D row-6D rotations.
        quat = matrix_to_quaternion(rotation_6d_to_matrix(rows)).numpy().astype(np.float32)
    else:
        raise ValueError("Prediction NPZ needs root_rot_6d")
    if root.ndim != 2 or root.shape[-1] != 3:
        raise ValueError(f"Invalid root shape: {root.shape}")
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError(f"Invalid root quaternion shape: {quat.shape}")
    if len(root) != len(quat) or len(root) != len(dof):
        raise ValueError("root, rotation, and dof lengths do not match")
    return root, quat, dof


def acceleration_error_mm_per_frame2(pred: np.ndarray, target: np.ndarray) -> float:
    total = min(len(pred), len(target))
    if total < 3:
        return float("nan")
    pred = pred[:total]
    target = target[:total]
    accel_pred = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    accel_target = target[2:] - 2.0 * target[1:-1] + target[:-2]
    return float(np.linalg.norm(accel_pred - accel_target, axis=-1).mean() * 1000.0)


def prediction_jitter_10mps3(pred: np.ndarray, fps: float = 30.0) -> float:
    if len(pred) < 4:
        return float("nan")
    jerk = (pred[3:] - 3.0 * pred[2:-1] + 3.0 * pred[1:-2] - pred[:-3]) * (float(fps) ** 3)
    return float(np.linalg.norm(jerk, axis=-1).mean(axis=-1).mean() / 10.0)


def root_traj_errors(pred_root: np.ndarray, target_root: np.ndarray) -> Dict[str, float]:
    total = min(len(pred_root), len(target_root))
    if total <= 0:
        return {"RTE": float("nan")}
    pred_root = pred_root[:total]
    target_root = target_root[:total]
    rigid_aligned = rigid_align_trajectory(pred_root, target_root)
    rigid_aligned_err = np.linalg.norm(rigid_aligned - target_root, axis=-1)
    return {"RTE": float(rigid_aligned_err.mean() * 1000.0)}


def rigid_align_trajectory(pred_root: np.ndarray, target_root: np.ndarray) -> np.ndarray:
    """Align pred_root to target_root with one fixed-scale rigid transform."""
    pred = np.asarray(pred_root, dtype=np.float64)
    target = np.asarray(target_root, dtype=np.float64)
    if pred.shape[0] < 2:
        return pred.astype(np.float32)

    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_zero = pred - pred_mean
    target_zero = target - target_mean
    cov = (target_zero.T @ pred_zero) / max(1, pred.shape[0])
    u, _, vh = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh.T) < 0.0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vh
    trans = target_mean - rot @ pred_mean
    return (pred @ rot.T + trans).astype(np.float32)


def _contact_segments(contact_mask: np.ndarray):
    segments = []
    active = contact_mask.astype(bool)
    start = None
    for idx, flag in enumerate(np.concatenate([active, [False]])):
        if flag and start is None:
            start = idx
        elif start is not None and not flag:
            segments.append((start, idx))
            start = None
    return segments


def foot_metrics_from_series(foot_series: np.ndarray, contact: np.ndarray, height_axis: int = 1) -> Dict[str, float]:
    if foot_series is None or contact is None:
        return {
            "Viol": float("nan"),
        }
    total = min(len(foot_series), len(contact))
    foot = foot_series[:total]
    contact = np.asarray(contact[:total], dtype=np.float32)
    if contact.ndim == 1:
        contact = contact[:, None]
    foot_count = min(foot.shape[1], contact.shape[1])
    foot = foot[:, :foot_count]
    contact = contact[:, :foot_count] > 0.5
    horizontal_axes = [axis for axis in range(3) if axis != int(height_axis)]

    sliding_vals = []
    float_vals = []
    pen_vals = []
    height_vals = []
    var_vals = []
    for dim in range(foot_count):
        segs = _contact_segments(contact[:, dim])
        if not segs:
            continue
        ground = np.median(foot[contact[:, dim], dim, height_axis]) if np.any(contact[:, dim]) else np.median(foot[:, dim, height_axis])
        for start, end in segs:
            if end - start < 2:
                continue
            traj = foot[start:end, dim]
            delta = traj[:, height_axis] - ground
            height_vals.extend(np.abs(delta).tolist())
            var_vals.extend(np.abs(np.diff(traj[:, height_axis])).tolist())
            sliding_vals.extend(np.linalg.norm(traj[1:, horizontal_axes] - traj[:-1, horizontal_axes], axis=-1).tolist())
            float_vals.extend((delta > 0.02).astype(np.float32).tolist())
            pen_vals.extend((delta < -0.03).astype(np.float32).tolist())
    return {
        "Viol": float((np.mean(float_vals) + np.mean(pen_vals)) * 100.0)
        if (float_vals or pen_vals) else float("nan"),
    }


def foot_sliding_support_xy_mm(
    foot_series: np.ndarray,
    support_mask: np.ndarray,
    height_axis: int = 1,
) -> Dict[str, float]:
    """Measure horizontal displacement on externally supplied stable-support pairs.

    The mask must come from a reference label or motion rather than the
    prediction being evaluated, so fast-sliding predictions cannot exclude
    their own errors through a prediction-derived velocity gate.
    """
    if foot_series is None or support_mask is None:
        return {"FS": float("nan"), "foot_support_pair_count": 0}
    total = min(len(foot_series), len(support_mask))
    foot = np.asarray(foot_series[:total], dtype=np.float32)
    support = np.asarray(support_mask[:total], dtype=bool)
    foot_count = min(foot.shape[1], support.shape[1])
    if total < 2 or foot_count <= 0:
        return {"FS": float("nan"), "foot_support_pair_count": 0}
    horizontal_axes = [axis for axis in range(3) if axis != int(height_axis)]
    steps = np.linalg.norm(
        foot[1:, :foot_count, horizontal_axes] - foot[:-1, :foot_count, horizontal_axes],
        axis=-1,
    )
    stable_pairs = support[1:, :foot_count] & support[:-1, :foot_count]
    values = steps[stable_pairs]
    return {
        "FS": float(np.mean(values) * 1000.0) if values.size else float("nan"),
        "foot_support_pair_count": int(values.size),
    }


def shared_robot_positions(raw_path: Path, spec, chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(raw_path, allow_pickle=False) as data:
        root, quat, dof = load_motion_arrays(data)
    kin = RobotKinematicsModel(spec.xml_path, device="cpu", model_to_xml_dof=spec.model_to_xml_dof)
    link_indices = [kin.body_names.index(item.robot_link) for item in spec.shared_joint_map]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(dof), chunk_size):
            end = min(len(dof), start + chunk_size)
            body, body_rot = kin.forward_kinematics(
                torch.from_numpy(root[start:end]),
                torch.from_numpy(quat[start:end][:, [1, 2, 3, 0]]),
                torch.from_numpy(dof[start:end]),
            )
            chunks.append(_shared_markers_from_body(body, body_rot, kin, spec).numpy())
    shared = np.concatenate(chunks, axis=0).astype(np.float32)
    alpha = getattr(spec, "torso_spine2_proxy_alpha", None)
    if alpha is not None:
        names = [item.name for item in spec.shared_joint_map]
        if "torso" in names and "left_shoulder" in names and "right_shoulder" in names:
            shared = apply_torso_spine2_proxy(
                torch.from_numpy(shared).unsqueeze(0),
                names.index("torso"),
                names.index("left_shoulder"),
                names.index("right_shoulder"),
                float(alpha),
            )[0].numpy()
    return shared, root, quat


def load_reference_root(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    smpl = np.load(sample_dir / "smpl.npy", allow_pickle=True).item()
    transl = np.asarray(smpl["transl"], dtype=np.float32)
    orient = np.asarray(smpl["global_orient"], dtype=np.float32)
    if orient.ndim == 3:
        orient = orient[:, 0]
    return transl.astype(np.float32), axis_angle_to_quat_wxyz(orient)


def smpl_robot_shaped_shared(sample_dir: Path, spec) -> np.ndarray:
    keypoints = np.load(sample_dir / "keypoints.npy").astype(np.float32)
    idx = [int(item.smpl_joint) for item in spec.shared_joint_map]
    shared = keypoints[:, idx]
    target = scale_shared_to_robot_shape(
        torch.from_numpy(shared).unsqueeze(0),
        torch.tensor(spec.gamp_edge_scale, dtype=torch.float32),
        parents=spec.shared_joint_parents,
    )[0].numpy()
    alpha = getattr(spec, "torso_spine2_proxy_alpha", None)
    if alpha is not None:
        names = [item.name for item in spec.shared_joint_map]
        if "torso" in names and "left_shoulder" in names and "right_shoulder" in names:
            target = apply_torso_spine2_proxy(
                torch.from_numpy(target).unsqueeze(0),
                names.index("torso"),
                names.index("left_shoulder"),
                names.index("right_shoulder"),
                float(alpha),
            )[0].numpy()
    return target.astype(np.float32)


def _direction_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean angle between corresponding vectors, in degrees."""
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    valid = (pred_norm[..., 0] > 1e-8) & (target_norm[..., 0] > 1e-8)
    if not np.any(valid):
        return float("nan")
    pred_unit = pred / np.clip(pred_norm, 1e-8, None)
    target_unit = target / np.clip(target_norm, 1e-8, None)
    cosine = np.clip(np.sum(pred_unit * target_unit, axis=-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine[valid])).mean())


def direction_metrics(pred_shared: np.ndarray, target_shared: np.ndarray, spec) -> Dict[str, float]:
    """Compute the paper's torso and end-effector direction errors."""
    names = [item.name for item in spec.shared_joint_map]
    out = {"TDE": float("nan"), "EDE": float("nan")}
    if "pelvis" in names and "torso" in names:
        pelvis, torso = names.index("pelvis"), names.index("torso")
        out["TDE"] = _direction_error_deg(pred_shared[:, torso] - pred_shared[:, pelvis],
                                            target_shared[:, torso] - target_shared[:, pelvis])
    if "pelvis" in names and all(name in names for name in ("left_wrist", "right_wrist")):
        pelvis = names.index("pelvis")
        values = []
        for wrist in ("left_wrist", "right_wrist"):
            idx = names.index(wrist)
            values.append(_direction_error_deg(pred_shared[:, idx] - pred_shared[:, pelvis],
                                                target_shared[:, idx] - target_shared[:, pelvis]))
        values = [value for value in values if np.isfinite(value)]
        if values:
            out["EDE"] = float(np.mean(values))
    return out


def load_support_labels(sample_dir: Path, total: int) -> np.ndarray | None:
    """Load optional external support/contact labels without guessing from predictions."""
    candidates = (
        "support.npy", "support_label.npy", "static_label.npy", "contact.npy",
        "contact_label.npy", "robot_contact.npy",
    )
    for name in candidates:
        path = sample_dir / name
        if not path.is_file():
            continue
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = next((key for key in ("support", "static_label", "contact", "contact_label") if key in loaded), None)
            if key is None:
                loaded.close()
                continue
            array = np.asarray(loaded[key])
            loaded.close()
        else:
            array = np.asarray(loaded)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] < total:
            continue
        # Static labels may be six channels in the fixed order
        # [left_ankle, left_foot, right_ankle, right_foot, left_wrist, right_wrist].
        if array.shape[1] >= 4:
            array = array[:, [1, 3]]
        elif array.shape[1] > 2:
            array = array[:, :2]
        return array[:total] > 0.5
    return None


def failure_rates(frame_mpjpe_mm: np.ndarray, window: int = 200) -> Dict[str, float | None]:
    """Compute Fail65/Fail100 only over complete paper-sized windows."""
    count = len(frame_mpjpe_mm) // int(window)
    if count <= 0:
        return {"Fail65": None, "Fail100": None, "evaluation_windows": 0}
    means = frame_mpjpe_mm[:count * window].reshape(count, window).mean(axis=1)
    return {
        "Fail65": float((means > 65.0).mean() * 100.0),
        "Fail100": float((means > 100.0).mean() * 100.0),
        "evaluation_windows": int(count),
    }


def gamp_smpl_metrics(pred_raw: Path, sample_dir: Path, spec, fps: float) -> Dict[str, float | None]:
    pred_shared, pred_root, _ = shared_robot_positions(pred_raw, spec)
    target = smpl_robot_shaped_shared(sample_dir, spec)
    ref_root, _ = load_reference_root(sample_dir)
    total = min(len(pred_shared), len(target), len(pred_root), len(ref_root))
    if total <= 0:
        raise ValueError(f"No overlapping frames for {pred_raw} and {sample_dir}")
    pred_shared = pred_shared[:total]
    target = target[:total]
    pred_root, ref_root = pred_root[:total], ref_root[:total]
    pa = pa_align_sequence(torch.from_numpy(pred_shared).unsqueeze(0),
                           torch.from_numpy(target).unsqueeze(0))[0].numpy()
    frame_mpjpe_mm = np.linalg.norm(pa - target, axis=-1).mean(axis=-1) * 1000.0
    row: Dict[str, float | None] = {
        "frames": int(total),
        "MAMPJPE": float(frame_mpjpe_mm.mean()),
        "Jitter": prediction_jitter_10mps3(pred_shared, fps=fps),
        "Accel": acceleration_error_mm_per_frame2(pred_shared, target),
        "FS": None,
        "Viol": None,
        # Execution metrics require a simulator trace and cannot be inferred
        # from an offline prediction NPZ.
        "Exec. SR": None,
        "Exec. MAMPJPE": None,
    }
    row.update(root_traj_errors(pred_root, ref_root))
    row.update(direction_metrics(pred_shared, target, spec))
    row.update(failure_rates(frame_mpjpe_mm))

    support = load_support_labels(sample_dir, total)
    if support is not None:
        names = [item.name for item in spec.shared_joint_map]
        foot_ids = [names.index(name) for name in ("left_foot", "right_foot") if name in names]
        if len(foot_ids) == 2:
            foot = pred_shared[:, foot_ids]
            row["FS"] = foot_sliding_support_xy_mm(foot, support, height_axis=1)["FS"]
            row["Viol"] = foot_metrics_from_series(foot, support, height_axis=1)["Viol"]
    for key in ("MAMPJPE", "RTE", "TDE", "EDE", "Jitter", "Accel", "FS", "Viol"):
        if key in row:
            row[key] = finite_or_none(row[key])
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, float | None]:
    if not rows:
        return {}
    weights = np.asarray([int(row.get("frames", 1)) for row in rows], dtype=np.float64)
    out = {"sequences": len(rows), "frames": int(weights.sum())}
    for key in sorted({key for row in rows for key in row}):
        if key in {"seq_name", "frames"}:
            continue
        vals = np.asarray([
            float(row[key]) if row.get(key) is not None else np.nan
            for row in rows
        ], dtype=np.float64)
        valid = np.isfinite(vals)
        if valid.any():
            out[key] = float(np.average(vals[valid], weights=weights[valid]))
        else:
            out[key] = None
    return out


def discover_prediction_dirs(pred_root: Path) -> List[Path]:
    return sorted({path.parent.parent for path in pred_root.rglob("*_raw_pred.npz")})


def discover_robot_predictions(sequence_dir: Path) -> Dict[str, Path]:
    predictions: Dict[str, Path] = {}
    for path in sequence_dir.glob("*/*_raw_pred.npz"):
        if path.is_file():
            predictions[path.parent.name] = path
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported robot prediction npz files.")
    parser.add_argument("--pred_root", required=True, help="predictions root containing <sequence>/<robot>/*_raw_pred.npz")
    parser.add_argument("--gt_root", required=True, help="Reference sequence root containing keypoints.npy and smpl.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    for config_path in sorted((PROJECT_ROOT / "configs").glob("robot_*.yaml")):
        try:
            spec = load_robot_spec(str(config_path), project_root=str(PROJECT_ROOT))
        except Exception as exc:
            print(f"warning: skipping invalid robot config {config_path}: {exc}", file=sys.stderr)
            continue
        specs[spec.name] = spec
    rows_by_robot: Dict[str, List[Dict[str, object]]] = {name: [] for name in specs}
    for seq_dir in discover_prediction_dirs(pred_root):
        rel = seq_dir.relative_to(pred_root)
        gt_seq = gt_root / rel
        if not gt_seq.exists():
            continue
        if not (gt_seq / "keypoints.npy").exists() or not (gt_seq / "smpl.npy").exists():
            continue
        for robot_name, prediction in discover_robot_predictions(seq_dir).items():
            if robot_name not in specs:
                print(f"warning: no robot config for {robot_name}; skipping {prediction}", file=sys.stderr)
                continue
            row = {"seq_name": str(rel)}
            row.update(gamp_smpl_metrics(prediction, gt_seq, specs[robot_name], args.fps))
            rows_by_robot[robot_name].append(row)
    reports = {
        f"gt_smpl_vs_{robot}_pred": {"aggregate": aggregate(rows), "per_sequence": rows}
        for robot, rows in rows_by_robot.items() if rows
    }
    summary_rows = [{"eval_group": key, **value["aggregate"]} for key, value in reports.items()]
    summary = {
        "metric_units": PAPER_METRIC_UNITS,
        "metric_definitions": PAPER_METRIC_DEFINITIONS,
        "reports": reports,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "metrics_average_table.csv", summary_rows)
    for robot, rows in rows_by_robot.items():
        if rows:
            write_csv(out_dir / f"gt_smpl_vs_{robot}_pred.csv", rows)
    print(json.dumps({key: value["aggregate"] for key, value in reports.items()}, ensure_ascii=False, indent=2))
    print(f"saved robot evaluation: {out_dir}")


if __name__ == "__main__":
    main()
