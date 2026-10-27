# Ported from RGB2robo GVRMR f016326ff27977008a5dc61198806417df57ea26.
import re
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from lib.robot import torch_utils as robot_torch_utils
from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import RobotSpec
from lib.util.beyondretarget.g1_loss_utils import masked_reduce_mean_g, rot6d_geodesic_g, rot6d_to_quat_wxyz
from lib.util.beyondretarget.gamp_pa_mpjpe_utils import SHARED_16_PARENTS, pa_align_sequence, scale_shared_to_robot_shape, valid_shared


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_reduce_mean_g(F.smooth_l1_loss(pred, target, reduction="none"), mask)


def _normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / torch.linalg.norm(values, dim=-1, keepdim=True).clamp_min(eps)


def _fps_scale(values: torch.Tensor, fps: float | torch.Tensor) -> torch.Tensor:
    """Return FPS with singleton dimensions compatible with a batched sequence."""
    if torch.is_tensor(fps):
        if fps.ndim == 2:
            if not torch.allclose(fps, fps[:, :1].expand_as(fps)):
                raise ValueError('Variable FPS within a clip requires timestamps, not a scalar derivative')
            fps = fps[:, 0]
        if fps.numel() not in (1, values.shape[0]) or not torch.isfinite(fps).all() or not (fps > 0).all():
            raise ValueError('FPS must be positive and scalar or one value per sequence')
        return fps.to(device=values.device, dtype=values.dtype).reshape(-1, *([1] * (values.ndim - 1)))
    return values.new_tensor(float(fps))


def _velocity(values: torch.Tensor, fps: float | torch.Tensor) -> torch.Tensor:
    return (values[:, 1:] - values[:, :-1]) * _fps_scale(values, fps)


def _accel(values: torch.Tensor, fps: float | torch.Tensor) -> torch.Tensor:
    scale = _fps_scale(values, fps)
    return (values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]) * scale * scale


def _second_diff(values: torch.Tensor) -> torch.Tensor:
    return values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]


def _jerk(values: torch.Tensor, fps: float) -> torch.Tensor:
    return (
        values[:, 3:]
        - 3.0 * values[:, 2:-1]
        + 3.0 * values[:, 1:-2]
        - values[:, :-3]
    ) * float(fps) * float(fps) * float(fps)


def _third_diff(values: torch.Tensor) -> torch.Tensor:
    return values[:, 3:] - 3.0 * values[:, 2:-1] + 3.0 * values[:, 1:-2] - values[:, :-3]


def _root_localize_sequence(shared: torch.Tensor, root_rot_6d: torch.Tensor) -> torch.Tensor:
    local = shared - shared[:, :, :1]
    root_quat_xyzw = rot6d_to_quat_wxyz(root_rot_6d)[..., [1, 2, 3, 0]]
    root_inv = robot_torch_utils.quat_conjugate(root_quat_xyzw.reshape(-1, 4))
    bsz, seq_len, joint_count, _ = local.shape
    local_flat = local.reshape(bsz * seq_len, joint_count, 3)
    local = robot_torch_utils.quat_rotate(
        root_inv[:, None, :].expand(-1, joint_count, -1).reshape(-1, 4),
        local_flat.reshape(-1, 3),
    ).reshape(bsz, seq_len, joint_count, 3)
    return local


class SmplSupervisedRobotLoss(nn.Module):
    DEFAULT_SMOOTH_ROOT_POS_SCALE = 0.00939802
    DEFAULT_SMOOTH_ROOT_ROT_SCALE = 0.00366108
    DEFAULT_SMOOTH_DOF_SCALE = 0.03387705

    def __init__(
        self,
        robot_spec: RobotSpec,
        weight: Dict[str, float],
        device,
        fps: float = 30.0,
        target_scale_mode: str = "gamp",
        use_torso_proxy: bool = False,
        robot_to_target_coord: str = "identity",
    ):
        super().__init__()
        self.robot_spec = robot_spec
        self.weight = dict(weight)
        self.fps = float(fps)
        self.target_scale_mode = str(target_scale_mode).lower()
        if self.target_scale_mode not in {"gamp", "pa"}:
            raise ValueError(f"Unsupported target_scale_mode: {target_scale_mode}")
        self.use_torso_proxy = bool(use_torso_proxy)
        self.robot_to_target_coord = str(robot_to_target_coord).lower()
        if self.robot_to_target_coord == "identity":
            robot_to_target_rotation = torch.eye(3, dtype=torch.float32)
        elif self.robot_to_target_coord == "zup_to_yup":
            # World Y-up/Z-up conversion, NOT the robot-local body basis.
            # G1 local axes are forward +X, left +Y, up +Z.
            robot_to_target_rotation = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported robot_to_target_coord: {robot_to_target_coord}")
        self.register_buffer("robot_to_target_rotation", robot_to_target_rotation)
        self.kinematics = RobotKinematicsModel(
            robot_spec.xml_path,
            device=str(device),
            model_to_xml_dof=robot_spec.model_to_xml_dof,
            neutral_dof=robot_spec.neutral_dof,
        )
        self.robot_link_indices = [self.kinematics.body_names.index(item.robot_link) for item in robot_spec.shared_joint_map]
        marker_offsets = []
        for item in robot_spec.shared_joint_map:
            if item.local_offset is None:
                marker_offsets.append([0.0, 0.0, 0.0])
            elif len(item.local_offset) != 3:
                raise ValueError(f"Expected local_offset with 3 values for {item.name}, got {item.local_offset}")
            else:
                marker_offsets.append([float(value) for value in item.local_offset])
        self.register_buffer("robot_marker_offsets", torch.as_tensor(marker_offsets, dtype=torch.float32))
        self.smpl_joint_indices = [int(item.smpl_joint) for item in robot_spec.shared_joint_map]
        self.shared_names = [item.name for item in robot_spec.shared_joint_map]
        self.torso_index = self.shared_names.index("torso") if "torso" in self.shared_names else -1
        self.left_shoulder_index = self.shared_names.index("left_shoulder") if "left_shoulder" in self.shared_names else -1
        self.right_shoulder_index = self.shared_names.index("right_shoulder") if "right_shoulder" in self.shared_names else -1
        tail_ids = robot_spec.tail_shared_indices or [3, 6, 10, 13]
        self.tail_shared_indices = [int(idx) for idx in tail_ids]
        self.foot_shared_indices = [
            idx for idx, name in enumerate(self.shared_names)
            if "foot" in name or "toe" in name
        ]
        if len(self.foot_shared_indices) < 2:
            self.foot_shared_indices = [
                idx for idx, name in enumerate(self.shared_names)
                if "ankle" in name
            ]
        if len(self.foot_shared_indices) < 2:
            self.foot_shared_indices = [4, 8] if len(self.shared_names) > 8 else list(range(min(2, len(self.shared_names))))
        self.wrist_dir_pairs = self._build_direction_pairs(
            (
                ("left_wrist", "left_hand"),
                ("right_wrist", "right_hand"),
            ),
            enabled=self._has_any_model_dof(("wrist", "hand_joint", "wry", "wrx", "gripper")),
        )
        self.ankle_dir_pairs = self._build_direction_pairs(
            (
                ("left_ankle", "left_foot"),
                ("right_ankle", "right_foot"),
            ),
            enabled=self._has_any_model_dof(("ankle", "_aky", "_akx", "aky", "akx")),
        )
        self.torso_dir_pairs = self._build_direction_pairs(
            (("pelvis", "torso"),),
            enabled="pelvis" in self.shared_names and "torso" in self.shared_names,
        )
        self.upper_arm_dir_pairs = self._build_direction_pairs(
            (
                ("left_shoulder", "left_elbow"),
                ("right_shoulder", "right_elbow"),
            ),
            enabled=True,
        )
        self.forearm_dir_pairs = self._build_direction_pairs(
            (
                ("left_elbow", "left_wrist"),
                ("right_elbow", "right_wrist"),
            ),
            enabled=True,
        )
        self.arm_segment_dir_pairs = self.upper_arm_dir_pairs + self.forearm_dir_pairs
        self.thigh_dir_pairs = self._build_direction_pairs(
            (
                ("left_hip", "left_knee"),
                ("right_hip", "right_knee"),
            ),
            enabled=True,
        )
        self.shank_dir_pairs = self._build_direction_pairs(
            (
                ("left_knee", "left_ankle"),
                ("right_knee", "right_ankle"),
            ),
            enabled=True,
        )
        self.leg_segment_dir_pairs = self.thigh_dir_pairs + self.shank_dir_pairs
        self.all_dir_pairs = (
            self.wrist_dir_pairs
            + self.ankle_dir_pairs
            + self.torso_dir_pairs
            + self.arm_segment_dir_pairs
            + self.leg_segment_dir_pairs
        )
        self.elbow_angle_triplets = self._build_angle_triplets(
            (
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ),
            enabled=True,
        )
        shared_count = len(robot_spec.shared_joint_map)
        default_parents = SHARED_16_PARENTS if shared_count == len(SHARED_16_PARENTS) else [-1] + list(range(shared_count - 1))
        self.shared_joint_parents = list(getattr(robot_spec, "shared_joint_parents", []) or default_parents)
        if len(self.shared_joint_parents) != shared_count:
            raise ValueError(
                f"Expected {shared_count} shared_joint_parents, got {len(self.shared_joint_parents)} for {robot_spec.name}"
            )
        se3_primary, se3_reference = self._build_se3_frame_recipes()
        self.register_buffer("se3_primary_indices", torch.as_tensor(se3_primary, dtype=torch.long), persistent=False)
        self.register_buffer("se3_reference_indices", torch.as_tensor(se3_reference, dtype=torch.long), persistent=False)
        edge_scale = list(getattr(robot_spec, "gamp_edge_scale", []) or [1.0] * shared_count)
        if len(edge_scale) != shared_count:
            raise ValueError(f"Expected {shared_count} gamp_edge_scale values, got {len(edge_scale)} for {robot_spec.name}")
        self.register_buffer("gamp_edge_scale", torch.as_tensor(edge_scale, dtype=torch.float32))
        self.gamp_edge_scale_initialized = True
        root_pos_share = float(self.weight.get("smooth_root_pos_share", 0.2))
        root_rot_share = float(self.weight.get("smooth_root_rot_share", 0.2))
        dof_share = float(self.weight.get("smooth_dof_share", 0.6))
        share_sum = max(root_pos_share + root_rot_share + dof_share, 1e-8)
        self.smooth_root_pos_share = root_pos_share / share_sum
        self.smooth_root_rot_share = root_rot_share / share_sum
        self.smooth_dof_share = dof_share / share_sum
        self.smooth_root_pos_scale = max(
            float(self.weight.get("smooth_root_pos_scale", self.DEFAULT_SMOOTH_ROOT_POS_SCALE)),
            1e-8,
        )
        self.smooth_root_rot_scale = max(
            float(self.weight.get("smooth_root_rot_scale", self.DEFAULT_SMOOTH_ROOT_ROT_SCALE)),
            1e-8,
        )
        self.smooth_dof_scale = max(
            float(self.weight.get("smooth_dof_scale", self.DEFAULT_SMOOTH_DOF_SCALE)),
            1e-8,
        )
        self.smooth_reference_scale = max(
            float(self.weight.get("smooth_reference_scale", self.smooth_dof_scale)),
            1e-8,
        )
        self.elbow_angle_loss_mode = str(self.weight.get("elbow_angle_loss_mode", "angle_flex")).lower()
        if self.elbow_angle_loss_mode not in {"angle_flex", "cosine"}:
            raise ValueError(f"Unsupported elbow_angle_loss_mode: {self.elbow_angle_loss_mode}")
        self.elbow_near_straight_flex_deg = max(
            float(self.weight.get("elbow_near_straight_flex_deg", 10.0)),
            0.0,
        )
        self.elbow_near_straight_weight = max(
            float(self.weight.get("elbow_near_straight_weight", 0.7)),
            0.0,
        )
        self.elbow_near_straight_ramp_low_deg = max(
            float(self.weight.get("elbow_near_straight_ramp_low_deg", 8.0)),
            0.0,
        )
        self.elbow_near_straight_ramp_high_deg = max(
            float(self.weight.get("elbow_near_straight_ramp_high_deg", 10.0)),
            self.elbow_near_straight_ramp_low_deg + 1e-6,
        )
        self.joint_limit_near_fraction = min(
            max(float(self.weight.get("joint_limit_near_fraction", 0.05)), 0.0),
            0.5,
        )
        self.joint_limit_margin_fraction = min(
            max(float(self.weight.get("joint_limit_margin_fraction", self.joint_limit_near_fraction)), 0.0),
            0.5,
        )

    def _has_any_model_dof(self, tokens: Iterable[str]) -> bool:
        names = [str(name).lower() for name in self.robot_spec.model_dof_order]
        return any(any(token in name for token in tokens) for name in names)

    def _build_direction_pairs(self, name_pairs: Iterable[Tuple[str, str]], enabled: bool) -> List[Tuple[int, int]]:
        if not enabled:
            return []
        pairs = []
        for start_name, end_name in name_pairs:
            if start_name in self.shared_names and end_name in self.shared_names:
                pairs.append((self.shared_names.index(start_name), self.shared_names.index(end_name)))
        return pairs

    def _build_angle_triplets(self, name_triplets: Iterable[Tuple[str, str, str]], enabled: bool) -> List[Tuple[int, int, int]]:
        if not enabled:
            return []
        triplets = []
        for start_name, center_name, end_name in name_triplets:
            if start_name in self.shared_names and center_name in self.shared_names and end_name in self.shared_names:
                triplets.append(
                    (
                        self.shared_names.index(start_name),
                        self.shared_names.index(center_name),
                        self.shared_names.index(end_name),
                    )
                )
        return triplets

    def _build_se3_frame_recipes(self) -> Tuple[List[int], List[int]]:
        """Choose a longitudinal and a cross-chain axis for every semantic point."""
        children = {index: [] for index in range(len(self.shared_names))}
        for index, parent in enumerate(self.shared_joint_parents):
            if parent >= 0:
                children[parent].append(index)

        def index_or(name: str, fallback: int) -> int:
            return self.shared_names.index(name) if name in self.shared_names else fallback

        pelvis = index_or("pelvis", 0)
        torso = index_or("torso", pelvis)
        left_hip = index_or("left_hip", pelvis)
        right_hip = index_or("right_hip", pelvis)
        left_shoulder = index_or("left_shoulder", torso)
        right_shoulder = index_or("right_shoulder", torso)
        primary: List[int] = []
        reference: List[int] = []
        for index, name in enumerate(self.shared_names):
            if name == "pelvis":
                axis, ref = torso, left_hip
            elif name == "torso":
                axis, ref = pelvis, left_shoulder
            else:
                axis = children[index][0] if children[index] else self.shared_joint_parents[index]
                if axis < 0:
                    axis = torso
                is_leg = any(token in name for token in ("hip", "knee", "ankle", "foot"))
                if name.startswith("left_"):
                    ref = right_hip if is_leg else right_shoulder
                elif name.startswith("right_"):
                    ref = left_hip if is_leg else left_shoulder
                else:
                    ref = left_shoulder
            if ref == index or ref == axis:
                ref = pelvis if pelvis not in {index, axis} else torso
            primary.append(axis)
            reference.append(ref)
        return primary, reference

    def _smooth_weight(self) -> float:
        return float(self.weight.get("smooth", self.weight.get("dof_accel_smooth", 0.0)))

    def _use_foot_contact_ground(self) -> bool:
        return float(self.weight.get("foot_contact_ground", 0.0)) > 0.0

    def _robot_bodies_native(self, root_rot_6d, dof):
        """Run FK in the native XML coordinate system, rooted at its origin."""
        leading_shape = root_rot_6d.shape[:-1]
        # The root orientation adapter is trainable. Run its 6D conversion and
        # the coupled FK branch in FP32: bf16 normalization near collinear 6D
        # axes can produce non-finite gradients before AMP can intervene.
        with torch.autocast(device_type=root_rot_6d.device.type, enabled=False):
            root_rotation_target = rotation_6d_to_matrix(root_rot_6d.float())
            conversion = self.robot_to_target_rotation.to(device=root_rot_6d.device, dtype=torch.float32)
            if self.robot_to_target_coord == "zup_to_yup":
                from lib.util.beyondretarget.root_basis import native_root_rotation
                root_rotation_native = native_root_rotation(root_rotation_target, conversion, self.robot_spec.name)
            else:
                root_rotation_native = conversion.transpose(-1, -2) @ root_rotation_target @ conversion
            root_rotation_native_6d = matrix_to_rotation_6d(root_rotation_native)
            flat_root_quat = rot6d_to_quat_wxyz(root_rotation_native_6d).reshape(-1, 4)[..., [1, 2, 3, 0]]
            flat_dof = dof.float().reshape(-1, self.robot_spec.dof)
            flat_root_pos = torch.zeros((flat_dof.shape[0], 3), dtype=torch.float32, device=dof.device)
            body_pos, body_rot = self.kinematics.forward_kinematics(flat_root_pos, flat_root_quat, flat_dof)
        body_pos = body_pos.reshape(*leading_shape, body_pos.shape[-2], 3)
        body_rot = body_rot.reshape(*leading_shape, body_rot.shape[-2], 4)
        return body_pos, body_rot

    def _native_points_to_target(self, points: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        conversion = self.robot_to_target_rotation.to(device=points.device, dtype=points.dtype)
        return points @ conversion.transpose(-1, -2) + root_pos.unsqueeze(-2)

    def _robot_bodies(self, root_pos, root_rot_6d, dof):
        body_pos_native, body_rot = self._robot_bodies_native(root_rot_6d, dof)
        return self._native_points_to_target(body_pos_native, root_pos), body_rot

    def _shared_robot(self, pred):
        body_pos_native, body_rot = self._robot_bodies_native(pred["root_rot_6d"], pred["dof"])
        shared = body_pos_native[..., self.robot_link_indices, :]
        offsets = self.robot_marker_offsets.to(dtype=shared.dtype, device=shared.device)
        if torch.any(offsets != 0):
            marker_rot = body_rot[..., self.robot_link_indices, :]
            flat_rot = marker_rot.reshape(-1, 4)
            flat_offsets = offsets.view(*([1] * (shared.dim() - 2)), offsets.shape[-2], 3)
            flat_offsets = flat_offsets.expand(*shared.shape[:-1], 3).reshape(-1, 3)
            rotated_offsets = robot_torch_utils.quat_rotate(flat_rot, flat_offsets).reshape_as(shared)
            shared = shared + rotated_offsets
        shared = self._native_points_to_target(shared, pred["root_pos"])
        if self.use_torso_proxy:
            from lib.util.beyondretarget.gamp_pa_mpjpe_utils import apply_torso_spine2_proxy

            if self.left_shoulder_index >= 0 and self.right_shoulder_index >= 0 and self.torso_index >= 0:
                shared = apply_torso_spine2_proxy(
                    shared,
                    self.torso_index,
                    self.left_shoulder_index,
                    self.right_shoulder_index,
                    float(getattr(self.robot_spec, "torso_spine2_proxy_alpha", 0.0)),
                )
        return shared

    def _shared_smpl(self, target):
        keypoints = target["smpl_joints"]
        if keypoints.shape[-2] <= max(self.smpl_joint_indices):
            raise ValueError(f"SMPL keypoints shape {keypoints.shape} cannot satisfy joints {self.smpl_joint_indices}")
        return keypoints[..., self.smpl_joint_indices, :]

    @staticmethod
    def _anchor_shared_root(target_shared: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        """Use the GVHMR camera-translation origin as the semantic robot root.

        The SMPL vertex-regressed pelvis is offset from ``transl``. Robot FK
        uses its pelvis body as the root origin, so comparing it directly to
        the regressed point produces a fixed, non-deployable global offset.
        """
        return target_shared - target_shared[..., :1, :] + root_pos.unsqueeze(-2)

    def _target_shared_gamp(self, target_shared: torch.Tensor) -> torch.Tensor:
        if self.target_scale_mode == "pa":
            return target_shared
        if not self.gamp_edge_scale_initialized:
            gamp_target = target_shared
        else:
            gamp_target = scale_shared_to_robot_shape(target_shared, self.gamp_edge_scale, parents=self.shared_joint_parents)
        if self.use_torso_proxy:
            from lib.util.beyondretarget.gamp_pa_mpjpe_utils import apply_torso_spine2_proxy

            if self.left_shoulder_index >= 0 and self.right_shoulder_index >= 0 and self.torso_index >= 0:
                gamp_target = apply_torso_spine2_proxy(
                    gamp_target,
                    self.torso_index,
                    self.left_shoulder_index,
                    self.right_shoulder_index,
                    float(getattr(self.robot_spec, "torso_spine2_proxy_alpha", 0.0)),
                )
        return gamp_target

    def _joint_limit_loss(self, dof: torch.Tensor) -> torch.Tensor:
        lower = self.kinematics.dof_lower_limits.to(dtype=dof.dtype, device=dof.device)
        upper = self.kinematics.dof_upper_limits.to(dtype=dof.dtype, device=dof.device)
        if self.robot_spec.model_to_xml_dof is not None:
            xml_indices = torch.as_tensor(self.robot_spec.model_to_xml_dof, dtype=torch.long, device=dof.device)
            lower = lower[xml_indices]
            upper = upper[xml_indices]
        lower = lower.view(*([1] * (dof.dim() - 1)), -1)
        upper = upper.view(*([1] * (dof.dim() - 1)), -1)
        return (F.relu(lower - dof).square() + F.relu(dof - upper).square()).mean()

    def _joint_limit_margin_loss(self, dof: torch.Tensor) -> torch.Tensor:
        lower = self.kinematics.dof_lower_limits.to(dtype=dof.dtype, device=dof.device)
        upper = self.kinematics.dof_upper_limits.to(dtype=dof.dtype, device=dof.device)
        if self.robot_spec.model_to_xml_dof is not None:
            xml_indices = torch.as_tensor(self.robot_spec.model_to_xml_dof, dtype=torch.long, device=dof.device)
            lower, upper = lower[xml_indices], upper[xml_indices]
        lower = lower.view(*([1] * (dof.dim() - 1)), -1)
        upper = upper.view(*([1] * (dof.dim() - 1)), -1)
        span = (upper - lower).clamp_min(1e-6)
        margin = torch.minimum((dof - lower) / span, (upper - dof) / span)
        return F.relu(self.joint_limit_margin_fraction - margin).square().mean()

    def _joint_limit_over_rate(self, dof: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        lower = self.kinematics.dof_lower_limits.to(dtype=dof.dtype, device=dof.device)
        upper = self.kinematics.dof_upper_limits.to(dtype=dof.dtype, device=dof.device)
        if self.robot_spec.model_to_xml_dof is not None:
            xml_indices = torch.as_tensor(self.robot_spec.model_to_xml_dof, dtype=torch.long, device=dof.device)
            lower = lower[xml_indices]
            upper = upper[xml_indices]
        lower = lower.view(*([1] * (dof.dim() - 1)), -1)
        upper = upper.view(*([1] * (dof.dim() - 1)), -1)
        over = ((dof < lower) | (dof > upper)).to(dtype=dof.dtype)
        mask = valid.to(dtype=dof.dtype).unsqueeze(-1)
        denom = torch.clamp(mask.sum() * float(dof.shape[-1]), min=1.0)
        return (over * mask).sum() / denom

    def _joint_limit_diagnostics(self, dof: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        lower = self.kinematics.dof_lower_limits.to(dtype=dof.dtype, device=dof.device)
        upper = self.kinematics.dof_upper_limits.to(dtype=dof.dtype, device=dof.device)
        if self.robot_spec.model_to_xml_dof is not None:
            xml_indices = torch.as_tensor(self.robot_spec.model_to_xml_dof, dtype=torch.long, device=dof.device)
            lower, upper = lower[xml_indices], upper[xml_indices]
        span = (upper - lower).clamp_min(1e-6)
        lower = lower.view(*([1] * (dof.dim() - 1)), -1)
        upper = upper.view(*([1] * (dof.dim() - 1)), -1)
        span = span.view(*([1] * (dof.dim() - 1)), -1)
        valid_mask = valid.to(dtype=dof.dtype).unsqueeze(-1)
        valid_frames = valid_mask.sum().clamp_min(1.0)
        near_lower = (dof <= lower + self.joint_limit_near_fraction * span).to(dtype=dof.dtype)
        near_upper = (dof >= upper - self.joint_limit_near_fraction * span).to(dtype=dof.dtype)
        normalized_margin = torch.minimum((dof - lower) / span, (upper - dof) / span)
        diagnostics = {
            "joint_limit_near_lower_rate": (near_lower * valid_mask).sum() / (valid_frames * float(dof.shape[-1])),
            "joint_limit_near_upper_rate": (near_upper * valid_mask).sum() / (valid_frames * float(dof.shape[-1])),
            "joint_limit_margin_mean": (normalized_margin * valid_mask).sum() / (valid_frames * float(dof.shape[-1])),
        }
        for index, name in enumerate(self.robot_spec.model_dof_order):
            safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_").lower()
            diagnostics[f"dof_{safe_name}_near_lower_rate"] = (near_lower[..., index] * valid).sum() / valid_frames
            diagnostics[f"dof_{safe_name}_near_upper_rate"] = (near_upper[..., index] * valid).sum() / valid_frames
        return diagnostics

    def _smooth_mag_loss(self, values: torch.Tensor, mask: torch.Tensor, diff_fn) -> torch.Tensor:
        if values.shape[1] < 2:
            return values.new_zeros(())
        diff = diff_fn(values)
        if diff.numel() == 0 or not torch.any(mask):
            return values.new_zeros(())
        mag = torch.linalg.norm(diff, dim=-1) if diff.shape[-1] > 1 else diff.squeeze(-1).abs()
        return masked_reduce_mean_g(mag, mask)

    def _motion_smooth_loss(self, pred: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_root_pos_smooth = self._smooth_mag_loss(pred["root_pos"], mask, _second_diff)
        loss_root_rot_smooth = self._smooth_mag_loss(pred["root_rot_6d"], mask, _second_diff)
        loss_dof_smooth = self._smooth_mag_loss(pred["dof"], mask, _second_diff)
        loss_smooth = self.smooth_reference_scale * (
            self.smooth_root_pos_share * loss_root_pos_smooth / self.smooth_root_pos_scale
            + self.smooth_root_rot_share * loss_root_rot_smooth / self.smooth_root_rot_scale
            + self.smooth_dof_share * loss_dof_smooth / self.smooth_dof_scale
        )
        return {
            "loss_smooth": loss_smooth,
            "loss_smooth_root_pos": loss_root_pos_smooth,
            "loss_smooth_root_rot": loss_root_rot_smooth,
            "loss_smooth_dof": loss_dof_smooth,
        }

    def _direction_loss(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if len(pairs) == 0:
            return pred_shared.new_zeros(())
        start_ids = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=pred_shared.device)
        end_ids = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=pred_shared.device)
        pred_dir = _normalize(pred_shared[..., end_ids, :] - pred_shared[..., start_ids, :])
        target_dir = _normalize(target_shared[..., end_ids, :] - target_shared[..., start_ids, :])
        loss = 1.0 - (pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
        return masked_reduce_mean_g(loss, valid)

    def _direction_error_deg(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if len(pairs) == 0:
            return pred_shared.new_zeros(())
        start_ids = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=pred_shared.device)
        end_ids = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=pred_shared.device)
        pred_dir = _normalize(pred_shared[..., end_ids, :] - pred_shared[..., start_ids, :])
        target_dir = _normalize(target_shared[..., end_ids, :] - target_shared[..., start_ids, :])
        deg = torch.rad2deg(torch.acos((pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)))
        return masked_reduce_mean_g(deg, valid)

    def _shared_se3_frames(self, shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct differentiable semantic frames from the 18 shared landmarks.

        Each frame uses a chain axis and a cross-chain reference axis.  The
        latter resolves roll, which a single direction vector cannot observe.
        Frames with nearly collinear axes are excluded from rotation reduction.
        """
        primary = self.se3_primary_indices.to(device=shared.device)
        reference = self.se3_reference_indices.to(device=shared.device)
        origin = shared
        axis_z_raw = shared[..., primary, :] - origin
        reference_raw = shared[..., reference, :] - origin
        axis_z = _normalize(axis_z_raw)
        axis_y_raw = torch.cross(axis_z, reference_raw, dim=-1)
        axis_y_norm = torch.linalg.norm(axis_y_raw, dim=-1)
        axis_y = axis_y_raw / axis_y_norm.unsqueeze(-1).clamp_min(1e-8)
        axis_x = _normalize(torch.cross(axis_y, axis_z, dim=-1))
        rotation = torch.stack((axis_x, axis_y, axis_z), dim=-1)
        valid = axis_y_norm > 1e-4
        return rotation, valid

    def _se3_rotation_values(self, pred_shared: torch.Tensor, target_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_rotation, pred_valid = self._shared_se3_frames(pred_shared)
        target_rotation, target_valid = self._shared_se3_frames(target_shared)
        with torch.autocast(device_type=pred_shared.device.type, enabled=False):
            relative = pred_rotation.float().transpose(-1, -2) @ target_rotation.float()
            trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
            cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle = torch.acos(cosine)
        return angle, pred_valid & target_valid

    def _se3_rotation_loss(self, pred_shared: torch.Tensor, target_shared: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle, frame_valid = self._se3_rotation_values(pred_shared, target_shared)
        mask = frame_valid & (valid > 0.5).unsqueeze(-1)
        if not torch.any(mask):
            return pred_shared.new_zeros(()), frame_valid.float().mean()
        return masked_reduce_mean_g(angle, mask), frame_valid.float().mean()

    def _direction_error_metrics(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the same semantic segments in one coordinate convention."""
        use_torso_dir = float(self.weight.get("torso_dir", 0.0)) > 0.0
        use_leg_segment_dir = float(self.weight.get("leg_segment_dir", 0.0)) > 0.0
        active_all_pairs = (
            self.wrist_dir_pairs
            + self.ankle_dir_pairs
            + self.upper_arm_dir_pairs
            + self.forearm_dir_pairs
        )
        if use_torso_dir:
            active_all_pairs += self.torso_dir_pairs
        if use_leg_segment_dir:
            active_all_pairs += self.leg_segment_dir_pairs
        pair_sets = {
            "wrist_dir_error_deg": self.wrist_dir_pairs,
            "ankle_dir_error_deg": self.ankle_dir_pairs,
            "wrist_ankle_dir_error_deg": self.wrist_dir_pairs + self.ankle_dir_pairs,
            "upper_arm_dir_error_deg": self.upper_arm_dir_pairs,
            "forearm_dir_error_deg": self.forearm_dir_pairs,
            "arm_segment_dir_error_deg": self.arm_segment_dir_pairs,
            "all_dir_error_deg": active_all_pairs,
        }
        # Disabled direction groups are omitted rather than reported as a
        # misleading zero.  The aggregate "all" diagnostic is retained.
        if use_torso_dir:
            pair_sets["torso_dir_error_deg"] = self.torso_dir_pairs
        if use_leg_segment_dir:
            pair_sets.update(
                {
                    "thigh_dir_error_deg": self.thigh_dir_pairs,
                    "shank_dir_error_deg": self.shank_dir_pairs,
                    "leg_segment_dir_error_deg": self.leg_segment_dir_pairs,
                }
            )
        return {
            f"{name}{suffix}": self._direction_error_deg(pred_shared, target_shared, valid, pairs)
            for name, pairs in pair_sets.items()
        }

    def _segment_down_error_deg(
        self,
        shared: torch.Tensor,
        valid: torch.Tensor,
        pairs: List[Tuple[int, int]],
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if len(pairs) == 0:
            return shared.new_zeros(())
        start_ids = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=shared.device)
        end_ids = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=shared.device)
        direction = _normalize(shared[..., end_ids, :] - shared[..., start_ids, :])
        down = direction.new_tensor([0.0, 0.0, -1.0])
        deg = torch.rad2deg(torch.acos((direction * down).sum(dim=-1).clamp(-1.0, 1.0)))
        reduce_mask = valid if mask is None else mask
        return masked_reduce_mean_g(deg, reduce_mask)

    def _angle_cosines(
        self,
        shared: torch.Tensor,
        triplets: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        start_ids = torch.as_tensor([triplet[0] for triplet in triplets], dtype=torch.long, device=shared.device)
        center_ids = torch.as_tensor([triplet[1] for triplet in triplets], dtype=torch.long, device=shared.device)
        end_ids = torch.as_tensor([triplet[2] for triplet in triplets], dtype=torch.long, device=shared.device)
        start_dir = _normalize(shared[..., start_ids, :] - shared[..., center_ids, :])
        end_dir = _normalize(shared[..., end_ids, :] - shared[..., center_ids, :])
        return (start_dir * end_dir).sum(dim=-1).clamp(-1.0, 1.0)

    def _angle_radians(
        self,
        shared: torch.Tensor,
        triplets: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        cosines = self._angle_cosines(shared, triplets)
        return torch.acos(cosines.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    def _angle_cosine_loss(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        triplets: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(triplets) == 0:
            return pred_shared.new_zeros(())
        pred_cos = self._angle_cosines(pred_shared, triplets)
        target_cos = self._angle_cosines(target_shared, triplets)
        loss = F.smooth_l1_loss(pred_cos, target_cos, reduction="none")
        return masked_reduce_mean_g(loss, valid)

    def _angle_flexion_loss(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        triplets: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(triplets) == 0:
            return pred_shared.new_zeros(())
        pred_angle = self._angle_radians(pred_shared, triplets)
        target_angle = self._angle_radians(target_shared, triplets)
        angle_loss = masked_reduce_mean_g(
            F.smooth_l1_loss(pred_angle, target_angle.detach(), reduction="none"),
            valid,
        )

        threshold = pred_angle.new_tensor(self.elbow_near_straight_flex_deg * 3.141592653589793 / 180.0)
        if self.elbow_near_straight_weight <= 0.0 or threshold <= 0.0:
            return angle_loss
        target_flex = 3.141592653589793 - target_angle.detach()
        pred_flex = 3.141592653589793 - pred_angle
        ramp_low = pred_angle.new_tensor(self.elbow_near_straight_ramp_low_deg * 3.141592653589793 / 180.0)
        ramp_high = pred_angle.new_tensor(self.elbow_near_straight_ramp_high_deg * 3.141592653589793 / 180.0)
        ramp = ((ramp_high - target_flex) / (ramp_high - ramp_low).clamp_min(1e-8)).clamp(0.0, 1.0)
        straight_mask = ramp * (valid > 0.5).to(dtype=pred_angle.dtype).unsqueeze(-1)
        if torch.sum(straight_mask) <= 0.0:
            return angle_loss
        flex_excess = F.relu(pred_flex - threshold)
        straight_loss = masked_reduce_mean_g(
            F.smooth_l1_loss(flex_excess, torch.zeros_like(flex_excess), reduction="none"),
            straight_mask,
        )
        return angle_loss + self.elbow_near_straight_weight * straight_loss

    def _elbow_angle_loss(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.elbow_angle_loss_mode == "cosine":
            return self._angle_cosine_loss(pred_shared, target_shared, valid, self.elbow_angle_triplets)
        return self._angle_flexion_loss(pred_shared, target_shared, valid, self.elbow_angle_triplets)

    def _angle_error_deg(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        triplets: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(triplets) == 0:
            return pred_shared.new_zeros(())
        pred_angle = torch.acos(self._angle_cosines(pred_shared, triplets))
        target_angle = torch.acos(self._angle_cosines(target_shared, triplets))
        deg = torch.rad2deg(torch.abs(pred_angle - target_angle))
        return masked_reduce_mean_g(deg, valid)

    def _arm_straight_down_metrics(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = pred_shared.new_zeros(())
        if len(self.elbow_angle_triplets) == 0:
            return {
                "elbow_near_straight_pred_flex_deg": zero,
                "elbow_near_straight_target_flex_deg": zero,
                "elbow_near_straight_excess_deg": zero,
                "elbow_near_straight_over_rate": zero,
                "arm_down_straight_frame_rate": zero,
                "arm_down_straight_pred_flex_deg": zero,
                "arm_down_straight_pred_down_error_deg": zero,
                "arm_down_straight_target_down_error_deg": zero,
            }

        pred_angle = self._angle_radians(pred_shared, self.elbow_angle_triplets)
        target_angle = self._angle_radians(target_shared, self.elbow_angle_triplets)
        pred_flex_deg = torch.rad2deg(torch.pi - pred_angle)
        target_flex_deg = torch.rad2deg(torch.pi - target_angle)
        valid_elbow = (valid > 0.5).to(dtype=pred_shared.dtype).unsqueeze(-1)
        near_mask = valid_elbow * (target_flex_deg < self.elbow_near_straight_flex_deg).to(dtype=pred_shared.dtype)
        threshold = pred_flex_deg.new_tensor(self.elbow_near_straight_flex_deg)
        near_excess = F.relu(pred_flex_deg - threshold)
        near_over = (pred_flex_deg > threshold).to(dtype=pred_shared.dtype)

        target_down = self._segment_down_error_values_deg(target_shared, self.arm_segment_dir_pairs)
        down_mask = near_mask
        if target_down is not None:
            down_pair_mask = (target_down < 35.0).to(dtype=pred_shared.dtype)
            if down_pair_mask.shape[-1] == len(self.arm_segment_dir_pairs):
                left_down = down_pair_mask[..., 0:1] * down_pair_mask[..., 2:3]
                right_down = down_pair_mask[..., 1:2] * down_pair_mask[..., 3:4]
                side_down = torch.cat([left_down, right_down], dim=-1)
                down_mask = near_mask * side_down

        return {
            "elbow_near_straight_pred_flex_deg": masked_reduce_mean_g(pred_flex_deg, near_mask),
            "elbow_near_straight_target_flex_deg": masked_reduce_mean_g(target_flex_deg, near_mask),
            "elbow_near_straight_excess_deg": masked_reduce_mean_g(near_excess, near_mask),
            "elbow_near_straight_over_rate": masked_reduce_mean_g(near_over, near_mask),
            "arm_down_straight_frame_rate": down_mask.sum() / (valid_elbow.sum() * float(down_mask.shape[-1])).clamp_min(1.0),
            "arm_down_straight_pred_flex_deg": masked_reduce_mean_g(pred_flex_deg, down_mask),
            "arm_down_straight_pred_down_error_deg": self._segment_down_error_deg(
                pred_shared,
                valid,
                self.arm_segment_dir_pairs,
                self._expand_side_mask_to_segment_mask(down_mask),
            ),
            "arm_down_straight_target_down_error_deg": self._segment_down_error_deg(
                target_shared,
                valid,
                self.arm_segment_dir_pairs,
                self._expand_side_mask_to_segment_mask(down_mask),
            ),
        }

    def _segment_down_error_values_deg(
        self,
        shared: torch.Tensor,
        pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if len(pairs) == 0:
            return None
        start_ids = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=shared.device)
        end_ids = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=shared.device)
        direction = _normalize(shared[..., end_ids, :] - shared[..., start_ids, :])
        down = direction.new_tensor([0.0, 0.0, -1.0])
        return torch.rad2deg(torch.acos((direction * down).sum(dim=-1).clamp(-1.0, 1.0)))

    def _expand_side_mask_to_segment_mask(self, side_mask: torch.Tensor) -> torch.Tensor:
        if len(self.arm_segment_dir_pairs) != 4 or side_mask.shape[-1] != 2:
            return side_mask
        return torch.stack(
            [side_mask[..., 0], side_mask[..., 1], side_mask[..., 0], side_mask[..., 1]],
            dim=-1,
        )

    def _foot_contact_ground_loss(
        self,
        pred_shared: torch.Tensor,
        target_shared: torch.Tensor,
        valid: torch.Tensor,
        contact_gt: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred_shared.shape[-2] == 0 or len(self.foot_shared_indices) == 0:
            return pred_shared.new_zeros(())
        foot_ids = torch.as_tensor(self.foot_shared_indices, dtype=torch.long, device=pred_shared.device)
        pred_feet = pred_shared[..., foot_ids, :]
        target_feet = target_shared[..., foot_ids, :]
        valid_mask = valid > 0.5
        if contact_gt is not None and contact_gt.shape[-1] >= 2:
            if contact_gt.shape[0] != valid.shape[0]:
                min_b = min(contact_gt.shape[0], valid.shape[0])
                pred_shared = pred_shared[:min_b]
                target_shared = target_shared[:min_b]
                pred_feet = pred_feet[:min_b]
                target_feet = target_feet[:min_b]
                valid = valid[:min_b]
                valid_mask = valid_mask[:min_b]
                contact_gt = contact_gt[:min_b]
            foot_count = min(pred_feet.shape[-2], int(contact_gt.shape[-1]))
            frame_contact_mask = valid_mask.unsqueeze(-1) & (contact_gt[:, :, :foot_count] > 0.5)
            pair_contact_mask = (
                valid_mask[:, 1:].unsqueeze(-1)
                & valid_mask[:, :-1].unsqueeze(-1)
                & (contact_gt[:, 1:, :foot_count] > 0.5)
                & (contact_gt[:, :-1, :foot_count] > 0.5)
            )
            target_height = target_feet[:, :, :foot_count, 2]
            pred_height = pred_feet[:, :, :foot_count, 2]
            ground_height = target_height.new_zeros((target_height.shape[0], foot_count))
            has_contact = torch.zeros_like(ground_height, dtype=torch.bool)
            for batch_idx in range(target_height.shape[0]):
                for foot_idx in range(foot_count):
                    contact_values = target_height[batch_idx, :, foot_idx][frame_contact_mask[batch_idx, :, foot_idx]]
                    if contact_values.numel() > 0:
                        ground_height[batch_idx, foot_idx] = torch.median(contact_values)
                        has_contact[batch_idx, foot_idx] = True
            frame_contact_mask = frame_contact_mask & has_contact.unsqueeze(1)
            pair_contact_mask = pair_contact_mask & has_contact.unsqueeze(1)
            if not torch.any(frame_contact_mask):
                return pred_shared.new_zeros(())
            loss_height = masked_reduce_mean_g(torch.abs(pred_height - ground_height.unsqueeze(1)), frame_contact_mask)
            if torch.any(pair_contact_mask):
                loss_height_var = masked_reduce_mean_g(torch.abs(pred_height[:, 1:] - pred_height[:, :-1]), pair_contact_mask)
            else:
                loss_height_var = pred_shared.new_zeros(())
            return loss_height + 0.5 * loss_height_var
        pair_mask = (valid[:, 1:] > 0.5) & (valid[:, :-1] > 0.5)
        target_disp = torch.linalg.norm(target_feet[:, 1:] - target_feet[:, :-1], dim=-1)
        contact_mask = pair_mask.unsqueeze(-1) & (target_disp < 0.01)
        if not torch.any(contact_mask):
            return pred_shared.new_zeros(())
        return masked_reduce_mean_g(torch.abs(pred_feet[:, 1:, :, 2] - target_feet[:, 1:, :, 2]), contact_mask)

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = target["frame_mask"]
        pred_shared = self._shared_robot(pred)
        target_shared = self._anchor_shared_root(self._shared_smpl(target), target["root_pos"])
        target_shared_gamp = self._target_shared_gamp(target_shared)

        loss_root_pos = _masked_smooth_l1(pred["root_pos"], target["root_pos"], valid)
        loss_root_rot = rot6d_geodesic_g(pred["root_rot_6d"], target["root_rot_6d"], valid)
        pred_shared_pa = pa_align_sequence(pred_shared, target_shared_gamp, detach_transform=True)
        loss_smpl_fk = _masked_smooth_l1(pred_shared_pa, target_shared_gamp, valid)
        # SE(3) position uses the same GAMP/PA semantic target as the legacy
        # keypoint objective; rotation uses frames built from all shared joints.
        loss_se3_position = loss_smpl_fk
        loss_se3_rotation, se3_frame_valid_rate = self._se3_rotation_loss(
            pred_shared_pa,
            target_shared_gamp,
            valid,
        )
        # Root-aligned FK keeps the predicted root orientation and all robot
        # joint geometry observable while removing only translation. Unlike PA,
        # it cannot hide a bad global pose behind a fitted rigid transform.
        pred_shared_root_aligned = pred_shared - pred["root_pos"].unsqueeze(-2)
        target_shared_root_aligned = target_shared_gamp - target["root_pos"].unsqueeze(-2)
        loss_smpl_root_aligned_fk = _masked_smooth_l1(
            pred_shared_root_aligned,
            target_shared_root_aligned,
            valid,
        )

        pair_mask = valid[:, 1:] * valid[:, :-1]
        sample_fps = target.get("fps", self.fps)
        loss_root_vel = _masked_smooth_l1(
            _velocity(pred["root_pos"], sample_fps),
            _velocity(target["root_pos"], sample_fps),
            pair_mask,
        )

        tail_mask = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
        pred_shared_local = _root_localize_sequence(pred_shared, pred["root_rot_6d"])
        target_shared_local = _root_localize_sequence(target_shared_gamp, target["root_rot_6d"])
        loss_tail_vel = _masked_smooth_l1(
            _velocity(pred_shared_local[:, :, self.tail_shared_indices], sample_fps),
            _velocity(target_shared_local[:, :, self.tail_shared_indices], sample_fps),
            pair_mask,
        )
        loss_tail_accel = _masked_smooth_l1(
            _accel(pred_shared_local[:, :, self.tail_shared_indices], sample_fps),
            _accel(target_shared_local[:, :, self.tail_shared_indices], sample_fps),
            tail_mask,
        )
        loss_joint_limit = self._joint_limit_loss(pred["dof"])
        loss_joint_limit_margin = self._joint_limit_margin_loss(pred["dof"])
        joint_limit_over_rate = self._joint_limit_over_rate(pred["dof"], valid)
        joint_limit_diagnostics = self._joint_limit_diagnostics(pred["dof"], valid)
        quad_mask = valid[:, 3:] * valid[:, 2:-1] * valid[:, 1:-2] * valid[:, :-3]
        smooth_losses = self._motion_smooth_loss(pred, tail_mask)
        loss_smooth = smooth_losses["loss_smooth"]
        loss_body_jitter = self._smooth_mag_loss(pred_shared_local, quad_mask, _third_diff)
        if self._use_foot_contact_ground():
            contact_gt = target.get("contact_lr")
            loss_foot_contact_ground = self._foot_contact_ground_loss(pred_shared, target_shared_gamp, valid, contact_gt=contact_gt)
        else:
            loss_foot_contact_ground = pred_shared.new_zeros(())
        direction_weights = (
            "wrist_ankle_dir",
            "torso_dir",
            "arm_segment_dir",
            "leg_segment_dir",
            "elbow_angle_dir",
        )
        use_direction_supervision = any(float(self.weight.get(key, 0.0)) > 0.0 for key in direction_weights)
        use_torso_dir = use_direction_supervision and float(self.weight.get("torso_dir", 0.0)) > 0.0
        use_leg_segment_dir = use_direction_supervision and float(self.weight.get("leg_segment_dir", 0.0)) > 0.0
        if use_direction_supervision:
            loss_wrist_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.wrist_dir_pairs)
            loss_ankle_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.ankle_dir_pairs)
            loss_wrist_ankle_dir = self._direction_loss(
                pred_shared,
                target_shared_gamp,
                valid,
                self.wrist_dir_pairs + self.ankle_dir_pairs,
            )
            loss_upper_arm_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.upper_arm_dir_pairs)
            loss_forearm_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.forearm_dir_pairs)
            loss_arm_segment_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.arm_segment_dir_pairs)
            loss_elbow_angle_dir = self._elbow_angle_loss(pred_shared, target_shared_gamp, valid)
            loss_torso_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.torso_dir_pairs) if use_torso_dir else pred_shared.new_zeros(())
            loss_thigh_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.thigh_dir_pairs) if use_leg_segment_dir else pred_shared.new_zeros(())
            loss_shank_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.shank_dir_pairs) if use_leg_segment_dir else pred_shared.new_zeros(())
            loss_leg_segment_dir = self._direction_loss(pred_shared, target_shared_gamp, valid, self.leg_segment_dir_pairs) if use_leg_segment_dir else pred_shared.new_zeros(())
        else:
            loss_wrist_ankle_dir = pred_shared.new_zeros(())
            loss_torso_dir = pred_shared.new_zeros(())
            loss_arm_segment_dir = pred_shared.new_zeros(())
            loss_leg_segment_dir = pred_shared.new_zeros(())
            loss_elbow_angle_dir = pred_shared.new_zeros(())

        base_total = (
            float(self.weight["root_pos"]) * loss_root_pos
            + float(self.weight["root_rot"]) * loss_root_rot
            + float(self.weight["smpl_fk"]) * loss_smpl_fk
            + float(self.weight.get("root_vel", 0.0)) * loss_root_vel
            + float(self.weight.get("tail_vel", 0.0)) * loss_tail_vel
            + float(self.weight.get("tail_accel", 0.0)) * loss_tail_accel
            + float(self.weight.get("joint_limit", 0.0)) * loss_joint_limit
            + float(self.weight.get("joint_limit_margin", 0.0)) * loss_joint_limit_margin
            + self._smooth_weight() * loss_smooth
            + float(self.weight.get("body_jitter", 0.0)) * loss_body_jitter
            + float(self.weight.get("foot_contact_ground", 0.0)) * loss_foot_contact_ground
            + float(self.weight.get("wrist_ankle_dir", 0.0)) * loss_wrist_ankle_dir
            + float(self.weight.get("torso_dir", 0.0)) * loss_torso_dir
            + float(self.weight.get("arm_segment_dir", 0.0)) * loss_arm_segment_dir
            + float(self.weight.get("leg_segment_dir", 0.0)) * loss_leg_segment_dir
            + float(self.weight.get("elbow_angle_dir", 0.0)) * loss_elbow_angle_dir
        )
        dynamic_terms = (
            ("smpl_root_aligned_fk", loss_smpl_root_aligned_fk, "smpl_root_aligned_target_fraction"),
            ("se3_position", loss_se3_position, "se3_position_target_fraction"),
            ("se3_rotation", loss_se3_rotation, "se3_rotation_target_fraction"),
        )
        fractions = [max(0.0, float(self.weight.get(fraction_key, 0.0))) for _, _, fraction_key in dynamic_terms]
        total_fraction = sum(fractions)
        if total_fraction >= 0.95:
            raise ValueError(f"Dynamic robot-loss fractions must sum to < 0.95, got {total_fraction:.3f}")
        dynamic_scale = base_total.detach() / max(1.0 - total_fraction, 1e-6)
        weighted_dynamic = {}
        effective_dynamic_weight = {}
        for (name, raw, _), fraction in zip(dynamic_terms, fractions):
            if fraction > 0.0:
                effective = dynamic_scale * fraction / raw.detach().clamp_min(1e-8)
            else:
                effective = raw.new_tensor(float(self.weight.get(name, 0.0)))
            weighted_dynamic[name] = effective * raw
            effective_dynamic_weight[name] = effective
        loss_smpl_root_aligned_fk_weighted = weighted_dynamic["smpl_root_aligned_fk"]
        loss_se3_position_weighted = weighted_dynamic["se3_position"]
        loss_se3_rotation_weighted = weighted_dynamic["se3_rotation"]
        total = base_total + sum(weighted_dynamic.values())
        stats = {
            "loss": total,
            "loss_root_pos": loss_root_pos,
            "loss_root_rot": loss_root_rot,
            "loss_smpl_fk": loss_smpl_fk,
            "loss_smpl_root_aligned_fk": loss_smpl_root_aligned_fk,
            "loss_se3_position": loss_se3_position,
            "loss_se3_rotation": loss_se3_rotation,
            "se3_frame_valid_rate": se3_frame_valid_rate,
            "loss_root_vel": loss_root_vel,
            "loss_tail_vel": loss_tail_vel,
            "loss_tail_accel": loss_tail_accel,
            "loss_joint_limit": loss_joint_limit,
            "loss_joint_limit_margin": loss_joint_limit_margin,
            "joint_limit_over_rate": joint_limit_over_rate,
            **joint_limit_diagnostics,
            "loss_smooth": loss_smooth,
            "loss_smooth_root_pos": smooth_losses["loss_smooth_root_pos"],
            "loss_smooth_root_rot": smooth_losses["loss_smooth_root_rot"],
            "loss_smooth_dof": smooth_losses["loss_smooth_dof"],
            "loss_body_jitter": loss_body_jitter,
            "loss_root_pos_weighted": float(self.weight["root_pos"]) * loss_root_pos,
            "loss_root_rot_weighted": float(self.weight["root_rot"]) * loss_root_rot,
            "loss_smpl_fk_weighted": float(self.weight["smpl_fk"]) * loss_smpl_fk,
            "loss_smpl_root_aligned_fk_weighted": loss_smpl_root_aligned_fk_weighted,
            "loss_se3_position_weighted": loss_se3_position_weighted,
            "loss_se3_rotation_weighted": loss_se3_rotation_weighted,
            "smpl_root_aligned_fk_effective_weight": effective_dynamic_weight["smpl_root_aligned_fk"].detach(),
            "se3_position_effective_weight": effective_dynamic_weight["se3_position"].detach(),
            "se3_rotation_effective_weight": effective_dynamic_weight["se3_rotation"].detach(),
            "loss_root_vel_weighted": float(self.weight.get("root_vel", 0.0)) * loss_root_vel,
            "loss_tail_vel_weighted": float(self.weight.get("tail_vel", 0.0)) * loss_tail_vel,
            "loss_tail_accel_weighted": float(self.weight.get("tail_accel", 0.0)) * loss_tail_accel,
            "loss_joint_limit_weighted": float(self.weight.get("joint_limit", 0.0)) * loss_joint_limit,
            "loss_joint_limit_margin_weighted": float(self.weight.get("joint_limit_margin", 0.0)) * loss_joint_limit_margin,
            "loss_smooth_weighted": self._smooth_weight() * loss_smooth,
            "loss_body_jitter_weighted": float(self.weight.get("body_jitter", 0.0)) * loss_body_jitter,
        }
        if use_direction_supervision:
            stats.update(
                {
                    "loss_wrist_dir": loss_wrist_dir,
                    "loss_ankle_dir": loss_ankle_dir,
                    "loss_wrist_ankle_dir": loss_wrist_ankle_dir,
                    "loss_upper_arm_dir": loss_upper_arm_dir,
                    "loss_forearm_dir": loss_forearm_dir,
                    "loss_arm_segment_dir": loss_arm_segment_dir,
                    "loss_elbow_angle_dir": loss_elbow_angle_dir,
                    "loss_wrist_ankle_dir_weighted": float(self.weight.get("wrist_ankle_dir", 0.0)) * loss_wrist_ankle_dir,
                    "loss_arm_segment_dir_weighted": float(self.weight.get("arm_segment_dir", 0.0)) * loss_arm_segment_dir,
                    "loss_elbow_angle_dir_weighted": float(self.weight.get("elbow_angle_dir", 0.0)) * loss_elbow_angle_dir,
                }
            )
        if use_torso_dir:
            stats["loss_torso_dir"] = loss_torso_dir
            stats["loss_torso_dir_weighted"] = float(self.weight.get("torso_dir", 0.0)) * loss_torso_dir
        if use_leg_segment_dir:
            stats.update(
                {
                    "loss_thigh_dir": loss_thigh_dir,
                    "loss_shank_dir": loss_shank_dir,
                    "loss_leg_segment_dir": loss_leg_segment_dir,
                    "loss_leg_segment_dir_weighted": float(self.weight.get("leg_segment_dir", 0.0)) * loss_leg_segment_dir,
                }
            )
        if self._use_foot_contact_ground():
            stats["loss_foot_contact_ground"] = loss_foot_contact_ground
            stats["loss_foot_contact_ground_weighted"] = float(self.weight.get("foot_contact_ground", 0.0)) * loss_foot_contact_ground
        return stats

    def compute_metrics(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = target["frame_mask"]
        pred_shared = self._shared_robot(pred)
        target_shared = self._anchor_shared_root(self._shared_smpl(target), target["root_pos"])
        target_shared_gamp = self._target_shared_gamp(target_shared)
        pred_valid, target_valid = valid_shared(pred_shared, target_shared_gamp, valid)
        with torch.no_grad():
            pa_aligned = pa_align_sequence(pred_valid, target_valid)
        pa_mpjpe = torch.linalg.norm(pa_aligned - target_valid, dim=-1).mean() if target_valid.numel() else pred_shared.sum() * 0.0
        with torch.no_grad():
            se3_pred_pa = pa_align_sequence(pred_shared, target_shared_gamp)
            se3_angle, se3_frame_valid = self._se3_rotation_values(se3_pred_pa, target_shared_gamp)
            se3_mask = se3_frame_valid & (valid > 0.5).unsqueeze(-1)
            se3_rotation_deg = (
                masked_reduce_mean_g(torch.rad2deg(se3_angle), se3_mask)
                if torch.any(se3_mask)
                else pred_shared.sum() * 0.0
            )

        pair_mask = valid[:, 1:] * valid[:, :-1]
        tail_mask = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
        sample_fps = target.get("fps", self.fps)
        pred_shared_local = _root_localize_sequence(pred_shared, pred["root_rot_6d"])
        target_shared_local = _root_localize_sequence(target_shared_gamp, target["root_rot_6d"])
        root_vel = torch.linalg.norm(_velocity(pred["root_pos"], sample_fps) - _velocity(target["root_pos"], sample_fps), dim=-1)
        tail_vel = torch.linalg.norm(
            _velocity(pred_shared_local[:, :, self.tail_shared_indices], sample_fps)
            - _velocity(target_shared_local[:, :, self.tail_shared_indices], sample_fps),
            dim=-1,
        )
        tail_acc = torch.linalg.norm(
            _accel(pred_shared_local[:, :, self.tail_shared_indices], sample_fps)
            - _accel(target_shared_local[:, :, self.tail_shared_indices], sample_fps),
            dim=-1,
        )
        global_fk = torch.linalg.norm(pred_shared - target_shared_gamp, dim=-1)
        root_aligned_fk = torch.linalg.norm(
            (pred_shared - pred["root_pos"].unsqueeze(-2))
            - (target_shared_gamp - target["root_pos"].unsqueeze(-2)),
            dim=-1,
        )
        pa_fk = torch.linalg.norm(pa_align_sequence(pred_shared, target_shared_gamp) - target_shared_gamp, dim=-1)
        quad_mask = valid[:, 3:] * valid[:, 2:-1] * valid[:, 1:-2] * valid[:, :-3]
        body_jitter_mag = torch.linalg.norm(_third_diff(pred_shared_local), dim=-1)
        metrics = {
            "smpl_root_pos_mae_m": masked_reduce_mean_g(torch.linalg.norm(pred["root_pos"] - target["root_pos"], dim=-1), valid),
            "smpl_root_rot_deg": rot6d_geodesic_g(pred["root_rot_6d"], target["root_rot_6d"], valid) * (180.0 / torch.pi),
            "smpl_fk_mae_m": masked_reduce_mean_g(pa_fk, valid),
            "smpl_fk_global_mae_m": masked_reduce_mean_g(global_fk, valid),
            "smpl_fk_root_aligned_mae_m": masked_reduce_mean_g(root_aligned_fk, valid),
            "smpl_gamp_pa_mpjpe_mm": pa_mpjpe * 1000.0,
            "smpl_gamp_root_aligned_mpjpe_mm": masked_reduce_mean_g(root_aligned_fk, valid) * 1000.0,
            "se3_position_mpjpe_mm": pa_mpjpe * 1000.0,
            "se3_rotation_deg": se3_rotation_deg,
            "se3_frame_valid_rate": se3_frame_valid.float().mean(),
            "root_vel_mae_m_s": masked_reduce_mean_g(root_vel, pair_mask),
            "tail_vel_mae_m_s": masked_reduce_mean_g(tail_vel, pair_mask),
            "tail_accel_mae_m_s2": masked_reduce_mean_g(tail_acc, tail_mask),
            "joint_limit": self._joint_limit_loss(pred["dof"]),
            "joint_limit_margin": self._joint_limit_margin_loss(pred["dof"]),
            "joint_limit_over_rate": self._joint_limit_over_rate(pred["dof"], valid),
            **self._joint_limit_diagnostics(pred["dof"], valid),
            "body_jitter_pred_m_frame3": masked_reduce_mean_g(body_jitter_mag, quad_mask),
        }
        direction_weights = ("wrist_ankle_dir", "torso_dir", "arm_segment_dir", "leg_segment_dir", "elbow_angle_dir")
        if any(float(self.weight.get(key, 0.0)) > 0.0 for key in direction_weights):
            metrics["elbow_angle_dir_error_deg"] = self._angle_error_deg(pred_shared, target_shared_gamp, valid, self.elbow_angle_triplets)
            metrics.update(self._direction_error_metrics(pred_shared, target_shared_gamp, valid))
            metrics.update(self._direction_error_metrics(pred_shared_local, target_shared_local, valid, suffix="_root_local"))
            metrics.update(self._direction_error_metrics(pa_align_sequence(pred_shared, target_shared_gamp), target_shared_gamp, valid, suffix="_pa_aligned"))
            metrics.update(self._arm_straight_down_metrics(pred_shared, target_shared_gamp, valid))
        return metrics
