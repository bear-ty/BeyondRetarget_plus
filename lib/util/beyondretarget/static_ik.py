"""Vendored GVRMR CCD helpers (2026-10-15); see docs/stage2_vendor_provenance.md."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import RobotSpec, load_robot_spec, model_dof_to_xml_dof
from pytorch3d.transforms import matrix_to_axis_angle as rotation_matrix_to_angle_axis


@dataclass
class FootIKConfig:
    contact_threshold: float = 0.6
    contact_exit_threshold: Optional[float] = 0.35
    min_segment: int = 4
    max_contact_gap: int = 5
    contact_fade: int = 4
    height_axis: int = 2
    enable_contact_motion_gate: bool = True
    contact_motion_speed_threshold_mps: float = 1.0
    static_threshold: float = 0.8
    static_exit_threshold: Optional[float] = 0.6
    fps: float = 30.0
    root_xy_smooth: int = 9
    root_xy_gain: float = 1.0
    root_xy_clip: float = 0.02
    enable_root_xyz_stabilize: bool = True
    root_xyz_xy_clip: float = 0.15
    root_xyz_z_gain: float = 0.75
    root_xyz_z_clip: float = 0.04
    root_xyz_fade: int = 6
    foot_target_smooth: int = 4
    preserve_root_xy: bool = False
    ik_steps: int = 2
    ccd_step_scale: float = 0.10
    ccd_threshold: float = 1e-3
    enable_ik: bool = True
    enable_ik_delta_smoothing: bool = True
    ik_delta_smooth_window: int = 5
    ik_delta_rate_limit: float = 0.18
    enable_contact_foot_flat: bool = True
    contact_foot_flat_gain: float = 0.6
    contact_foot_flat_max_step: float = 0.16
    contact_foot_flat_local_axis: int = 2


def _foot_joint_specs(robot_spec: RobotSpec, kinematics: RobotKinematicsModel) -> Tuple[List[int], List[List[float]]]:
    preferred_names = ["left_foot", "right_foot"]
    wanted = []
    offsets = []
    for name in preferred_names:
        for item in robot_spec.shared_joint_map:
            if item.name != name:
                continue
            try:
                wanted.append(kinematics.body_names.index(item.robot_link))
                offsets.append([0.0, 0.0, 0.0] if item.local_offset is None else [float(v) for v in item.local_offset])
            except ValueError:
                pass
            break
    if len(wanted) == 2:
        return wanted, offsets
    fallback = []
    fallback_offsets = []
    for item in robot_spec.shared_joint_map:
        if "foot" in item.name or "toe" in item.robot_link:
            try:
                fallback.append(kinematics.body_names.index(item.robot_link))
                fallback_offsets.append([0.0, 0.0, 0.0] if item.local_offset is None else [float(v) for v in item.local_offset])
            except ValueError:
                continue
    if len(fallback) >= 2:
        return fallback[:2], fallback_offsets[:2]
    raise RuntimeError(f"Could not resolve foot links for {robot_spec.name}")


def _foot_joint_indices(robot_spec: RobotSpec, kinematics: RobotKinematicsModel) -> List[int]:
    foot_ids, _ = _foot_joint_specs(robot_spec, kinematics)
    return foot_ids


def _marker_positions(
    body_pos: torch.Tensor,
    body_rot_xyzw: torch.Tensor,
    body_ids: Sequence[int],
    offsets: torch.Tensor,
) -> torch.Tensor:
    markers = body_pos[:, list(body_ids)]
    if offsets.numel() == 0 or not bool(torch.any(offsets != 0.0)):
        return markers
    marker_rot = _quat_xyzw_to_rotmat(body_rot_xyzw[:, list(body_ids)])
    rotated_offsets = torch.einsum("tfij,fj->tfi", marker_rot, offsets.to(device=markers.device, dtype=markers.dtype))
    return markers + rotated_offsets


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[-1])], axis=-1)
    return smoothed.astype(np.float32)


def _quat_xyzw_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"expected quaternion [...,4], got {quat.shape}")
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _rotmat_to_quat_xyzw(rot: torch.Tensor) -> torch.Tensor:
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    q = torch.zeros(rot.shape[:-2] + (4,), dtype=rot.dtype, device=rot.device)
    cond = trace > 0.0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2.0
    q[..., 3] = 0.25 * s
    q[..., 0] = (rot[..., 2, 1] - rot[..., 1, 2]) / s.clamp(min=1e-8)
    q[..., 1] = (rot[..., 0, 2] - rot[..., 2, 0]) / s.clamp(min=1e-8)
    q[..., 2] = (rot[..., 1, 0] - rot[..., 0, 1]) / s.clamp(min=1e-8)
    if not torch.all(cond):
        alt = torch.zeros_like(q)
        diag = torch.stack([rot[..., 0, 0], rot[..., 1, 1], rot[..., 2, 2]], dim=-1)
        max_idx = torch.argmax(diag, dim=-1)
        for idx in range(3):
            mask = max_idx == idx
            if not torch.any(mask):
                continue
            if idx == 0:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2], min=1e-8)) * 2.0
                alt[..., 0] = 0.25 * s
                alt[..., 1] = (rot[..., 0, 1] + rot[..., 1, 0]) / s.clamp(min=1e-8)
                alt[..., 2] = (rot[..., 0, 2] + rot[..., 2, 0]) / s.clamp(min=1e-8)
                alt[..., 3] = (rot[..., 2, 1] - rot[..., 1, 2]) / s.clamp(min=1e-8)
            elif idx == 1:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2], min=1e-8)) * 2.0
                alt[..., 0] = (rot[..., 0, 1] + rot[..., 1, 0]) / s.clamp(min=1e-8)
                alt[..., 1] = 0.25 * s
                alt[..., 2] = (rot[..., 1, 2] + rot[..., 2, 1]) / s.clamp(min=1e-8)
                alt[..., 3] = (rot[..., 0, 2] - rot[..., 2, 0]) / s.clamp(min=1e-8)
            else:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1], min=1e-8)) * 2.0
                alt[..., 0] = (rot[..., 0, 2] + rot[..., 2, 0]) / s.clamp(min=1e-8)
                alt[..., 1] = (rot[..., 1, 2] + rot[..., 2, 1]) / s.clamp(min=1e-8)
                alt[..., 2] = 0.25 * s
                alt[..., 3] = (rot[..., 1, 0] - rot[..., 0, 1]) / s.clamp(min=1e-8)
            q = torch.where(mask[..., None], alt, q)
    return torch.nn.functional.normalize(q, dim=-1)


def _quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def _quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def _axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-8)
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)[..., 0]
    s = torch.sin(angle)[..., 0]
    one_c = 1.0 - c
    rot = torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], dim=-1),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], dim=-1),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    rot = torch.where((angle[..., 0] < 1e-8)[..., None, None], eye, rot)
    return rot


def _rotmat_from_two_vectors(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    src_norm = src / src.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dst_norm = dst / dst.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.cross(src_norm, dst_norm, dim=-1)
    c = torch.sum(src_norm * dst_norm, dim=-1, keepdim=True)
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    eye = torch.eye(3, device=src.device, dtype=src.dtype)
    vx = torch.zeros(src.shape[:-1] + (3, 3), device=src.device, dtype=src.dtype)
    vx[..., 0, 1] = -v[..., 2]
    vx[..., 0, 2] = v[..., 1]
    vx[..., 1, 0] = v[..., 2]
    vx[..., 1, 2] = -v[..., 0]
    vx[..., 2, 0] = -v[..., 1]
    vx[..., 2, 1] = v[..., 0]
    rot = eye + vx + vx @ vx * ((1.0 - c) / s.pow(2).clamp(min=1e-8))
    parallel = s[..., 0] < 1e-8
    same = parallel & (c[..., 0] > 0.0)
    opposite = parallel & ~same
    if torch.any(same):
        rot = torch.where(same[..., None, None], eye, rot)
    if torch.any(opposite):
        fallback_axis = torch.zeros_like(src_norm)
        fallback_axis[..., 0] = 1.0
        alt = _axis_angle_to_rotmat(np.pi * fallback_axis)
        rot = torch.where(opposite[..., None, None], alt, rot)
    return rot


def _build_joint_to_dof_lookup(kin: RobotKinematicsModel) -> Dict[int, int]:
    lookup = {}
    for joint_idx, joint in enumerate(kin._joints):  # noqa: SLF001
        if joint.dof_idx >= 0:
            lookup[joint_idx] = joint.dof_idx
    return lookup


def _build_ik_joint_whitelist(kin: RobotKinematicsModel, foot_ids: Sequence[int]) -> set[int]:
    whitelist: set[int] = set()
    for foot_id in foot_ids:
        chain = _chain_from_end_effector(kin._parent_indices, foot_id)  # noqa: SLF001
        whitelist.update(int(j) for j in chain[1:-1])
    return whitelist


def _chain_from_end_effector(parent_indices: torch.Tensor, end_effector: int) -> List[int]:
    chain = [int(end_effector)]
    curr = int(end_effector)
    while curr >= 0:
        curr = int(parent_indices[curr].item())
        if curr >= 0:
            chain.append(curr)
    return list(reversed(chain))


def _joint_local_rot_from_dof(joint, dof_val: torch.Tensor) -> torch.Tensor:
    if joint.dof_dim == 0:
        return torch.eye(3, device=dof_val.device, dtype=dof_val.dtype)
    if joint.dof_dim == 1:
        axis = joint._axis.to(device=dof_val.device, dtype=dof_val.dtype)  # noqa: SLF001
        angle = dof_val[..., :1]
        axis = axis.view(*([1] * (angle.dim() - 1)), 3)
        return _axis_angle_to_rotmat(axis * angle)
    if joint.dof_dim == 3:
        return _axis_angle_to_rotmat(dof_val)
    raise ValueError(f"Unsupported joint dof_dim={joint.dof_dim}")


def _joint_dof_from_local_rot(joint, local_rot: torch.Tensor) -> torch.Tensor:
    if local_rot.dim() == 2:
        local_rot = local_rot.unsqueeze(0)
    aa = rotation_matrix_to_angle_axis(local_rot)
    if joint.dof_dim == 1:
        axis = joint._axis.to(device=local_rot.device, dtype=local_rot.dtype)  # noqa: SLF001
        angle = (aa * axis).sum(dim=-1, keepdim=True)
        return angle
    if joint.dof_dim == 3:
        return aa
    return aa.new_zeros((aa.shape[0], 0)) if aa.dim() > 1 else aa.new_zeros((0,))


def _joint_local_rot_to_dof(joint, local_rot: torch.Tensor) -> torch.Tensor:
    aa = rotation_matrix_to_angle_axis(local_rot)
    if joint.dof_dim == 1:
        axis = joint._axis.to(device=local_rot.device, dtype=local_rot.dtype)  # noqa: SLF001
        return (aa * axis).sum(dim=-1, keepdim=True)
    if joint.dof_dim == 3:
        return aa
    return aa.new_zeros((aa.shape[0], 0)) if aa.dim() > 1 else aa.new_zeros((0,))


def _fixed_rotmat_from_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = torch.nn.functional.normalize(quat_xyzw, dim=-1)
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _signed_angle_on_axis(axis_world: torch.Tensor, src_vec: torch.Tensor, dst_vec: torch.Tensor) -> torch.Tensor:
    axis_world = axis_world / axis_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    src_proj = src_vec - axis_world * (src_vec * axis_world).sum(dim=-1, keepdim=True)
    dst_proj = dst_vec - axis_world * (dst_vec * axis_world).sum(dim=-1, keepdim=True)
    src_norm = src_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dst_norm = dst_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    src_unit = src_proj / src_norm
    dst_unit = dst_proj / dst_norm
    cos_angle = torch.clamp((src_unit * dst_unit).sum(dim=-1), -1.0, 1.0)
    sin_angle = (axis_world * torch.cross(src_unit, dst_unit, dim=-1)).sum(dim=-1)
    return torch.atan2(sin_angle, cos_angle)


def _forward_kinematics_dof(
    kin: RobotKinematicsModel,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    body_pos, body_rot = kin.forward_kinematics(root_pos, root_rot_xyzw, dof)
    return body_pos, body_rot


def _smooth_ik_delta(
    source_dof: np.ndarray,
    refined_dof: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: FootIKConfig,
) -> np.ndarray:
    if not bool(cfg.enable_ik_delta_smoothing):
        return np.clip(refined_dof, lower[None, :], upper[None, :]).astype(np.float32)
    delta = np.asarray(refined_dof - source_dof, dtype=np.float32)
    if int(cfg.ik_delta_smooth_window) > 1 and delta.shape[0] >= 3:
        delta = _moving_average_1d(delta, int(cfg.ik_delta_smooth_window))
    rate_limit = float(cfg.ik_delta_rate_limit)
    if rate_limit > 0.0 and delta.shape[0] > 1:
        limited = delta.copy()
        for idx in range(1, limited.shape[0]):
            step = np.clip(limited[idx] - limited[idx - 1], -rate_limit, rate_limit)
            limited[idx] = limited[idx - 1] + step
        for idx in range(limited.shape[0] - 2, -1, -1):
            step = np.clip(limited[idx] - limited[idx + 1], -rate_limit, rate_limit)
            limited[idx] = limited[idx + 1] + step
        delta = limited
    return np.clip(source_dof + delta, lower[None, :], upper[None, :]).astype(np.float32)


def _ccd_solve_sequence(
    kin: RobotKinematicsModel,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof_xml: torch.Tensor,
    target_foot_pos: torch.Tensor,
    foot_ids: Sequence[int],
    foot_offsets: torch.Tensor,
    support_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor] = None,
    max_iter: int = 2,
    step_scale: float = 0.12,
    ccd_threshold: float = 1e-3,
    enable_contact_foot_flat: bool = False,
    contact_foot_flat_gain: float = 0.4,
    contact_foot_flat_max_step: float = 0.1,
    contact_foot_flat_local_axis: int = 2,
    height_axis: int = 2,
) -> torch.Tensor:
    refined = dof_xml.clone()
    parent_indices = kin._parent_indices  # noqa: SLF001
    joint_to_dof = _build_joint_to_dof_lookup(kin)
    whitelist = _build_ik_joint_whitelist(kin, foot_ids)
    support_mask = support_mask.bool()
    if support_weight is None:
        support_weight = support_mask.to(dtype=dof_xml.dtype)
    else:
        support_weight = support_weight.to(device=dof_xml.device, dtype=dof_xml.dtype)

    for _ in range(max_iter):
        body_pos, body_rot = _forward_kinematics_dof(kin, root_pos, root_rot_xyzw, refined)
        marker_pos = _marker_positions(body_pos, body_rot, foot_ids, foot_offsets)
        max_err = 0.0
        for foot_slot, foot_id in enumerate(foot_ids):
            active = support_mask[:, foot_slot]
            if not bool(active.any()):
                continue
            target_pos = target_foot_pos[:, foot_slot]
            err = torch.linalg.norm(marker_pos[active, foot_slot] - target_pos[active], dim=-1)
            if err.numel() > 0:
                max_err = max(max_err, float(err.max().item()))

            chain = _chain_from_end_effector(parent_indices, foot_id)
            if len(chain) <= 2:
                continue

            for chain_idx in range(len(chain) - 2, 0, -1):
                joint_id = chain[chain_idx]
                if int(joint_id) not in whitelist:
                    continue
                dof_idx = joint_to_dof.get(joint_id)
                if dof_idx is None:
                    continue

                joint = kin._joints[joint_id]  # noqa: SLF001
                joint_pos = body_pos[:, joint_id]
                end_pos = marker_pos[:, foot_slot]
                src_vec = end_pos - joint_pos
                dst_vec = target_pos - joint_pos
                valid = active & (torch.linalg.norm(src_vec, dim=-1) > 1e-8) & (torch.linalg.norm(dst_vec, dim=-1) > 1e-8)
                if not bool(valid.any()):
                    continue

                parent_id = int(parent_indices[joint_id].item())
                parent_rot = _quat_xyzw_to_rotmat(body_rot[:, parent_id])
                fixed_rot = _fixed_rotmat_from_quat_xyzw(kin._local_rotation[joint_id].to(device=parent_rot.device, dtype=parent_rot.dtype).unsqueeze(0))[0]  # noqa: SLF001
                chain_weight = float(chain_idx) / float(max(len(chain) - 1, 1))
                chain_weight = float(np.clip(chain_weight, 0.15, 1.0))
                if joint.dof_dim == 1:
                    axis = joint._axis.to(device=parent_rot.device, dtype=parent_rot.dtype)  # noqa: SLF001
                    axis_world = torch.einsum("tij,j->ti", parent_rot @ fixed_rot, axis)
                    angle = _signed_angle_on_axis(axis_world, src_vec, dst_vec)
                    step = float(step_scale) * chain_weight
                    weight = support_weight[:, foot_slot]
                    updated = refined[:, dof_idx] + step * angle * weight
                    lower = kin.dof_lower_limits[dof_idx]
                    upper = kin.dof_upper_limits[dof_idx]
                    refined[valid, dof_idx] = torch.clamp(updated[valid], lower, upper)
                elif joint.dof_dim == 3:
                    delta_rot = _rotmat_from_two_vectors(src_vec, dst_vec)
                    current_local = refined[:, dof_idx : dof_idx + 3]
                    current_local_rot = _joint_local_rot_from_dof(joint, current_local)
                    updated_local_rot = delta_rot @ current_local_rot
                    rel_rot = updated_local_rot @ current_local_rot.transpose(-1, -2)
                    rel_aa = rotation_matrix_to_angle_axis(rel_rot)
                    weight = support_weight[:, foot_slot].view(-1, 1)
                    partial_rot = _axis_angle_to_rotmat(rel_aa * chain_weight * weight) @ current_local_rot
                    new_dof = _joint_local_rot_to_dof(joint, partial_rot)
                    lower = kin.dof_lower_limits[dof_idx : dof_idx + 3]
                    upper = kin.dof_upper_limits[dof_idx : dof_idx + 3]
                    refined[valid, dof_idx : dof_idx + 3] = torch.clamp(new_dof[valid], lower, upper)

                body_pos, body_rot = _forward_kinematics_dof(kin, root_pos, root_rot_xyzw, refined)
                marker_pos = _marker_positions(body_pos, body_rot, foot_ids, foot_offsets)

        if enable_contact_foot_flat:
            raise ValueError('Foot flattening is not part of this static-only solver')

        if max_err < float(ccd_threshold):
            break
    return refined
