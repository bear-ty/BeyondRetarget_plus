"""Canonical SMPL-parameter to robot-root axis convention."""

import torch


ROOT_BASIS_VERSION = "smpl_parameter_axes_v2"
ROBOT_IN_SMPL = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
AUDITED_ROBOTS = frozenset({
    "g1", "r1", "gr1t1_without_hand", "gr2v3_8_7_dummy_hand",
    "h1_with_hand_without_hand", "tienkung", "t1_serial", "atlas_v4",
})


def root_basis_metadata(robot_name: str) -> str:
    if robot_name not in AUDITED_ROBOTS:
        raise ValueError(f"Robot local axes not audited: {robot_name!r}")
    return ROOT_BASIS_VERSION


def native_root_rotation(target_rotation: torch.Tensor, world_conversion: torch.Tensor,
                         robot_name: str = "g1") -> torch.Tensor:
    """Convert native SMPL root rows to the audited robot/world convention."""
    root_basis_metadata(robot_name)
    basis = target_rotation.new_tensor(ROBOT_IN_SMPL)
    return world_conversion.transpose(-1, -2) @ target_rotation @ basis
