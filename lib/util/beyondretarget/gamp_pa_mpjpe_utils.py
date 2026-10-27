from typing import Optional, Sequence, Tuple

import torch


SHARED_16_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 9, 13, 14]


def make_tensor(values: Sequence[float], device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=dtype, device=device)


def scale_shared_to_robot_shape(
    shared: torch.Tensor,
    edge_scale: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if shared.shape[-1] != 3:
        raise ValueError(f"Expected shared joints with shape [..., N, 3], got {tuple(shared.shape)}")
    joint_count = int(shared.shape[-2])
    parents = list(SHARED_16_PARENTS if parents is None else parents)
    if len(parents) != joint_count:
        raise ValueError(f"Expected {joint_count} shared parents, got {len(parents)}")
    edge_scale = edge_scale.to(dtype=shared.dtype, device=shared.device)
    if edge_scale.numel() != joint_count:
        raise ValueError(f"Expected {joint_count} GAMP edge scales, got {edge_scale.numel()}")
    scaled = [shared[..., 0, :]]
    for joint_idx in range(1, joint_count):
        parent_idx = int(parents[joint_idx])
        if parent_idx < 0 or parent_idx >= joint_idx:
            raise ValueError(f"Invalid parent index {parent_idx} for shared joint {joint_idx}")
        edge_vec = shared[..., joint_idx, :] - shared[..., parent_idx, :]
        scaled.append(scaled[parent_idx] + edge_scale[joint_idx] * edge_vec)
    return torch.stack(scaled, dim=-2)


def scale_shared16_to_robot_shape(
    shared: torch.Tensor,
    edge_scale: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    return scale_shared_to_robot_shape(shared, edge_scale, parents=parents)


def apply_torso_spine2_proxy(
    shared: torch.Tensor,
    torso_index: int,
    left_shoulder_index: int,
    right_shoulder_index: int,
    alpha: float,
) -> torch.Tensor:
    if shared.shape[-1] != 3:
        raise ValueError(f"Expected shared joints with shape [..., N, 3], got {tuple(shared.shape)}")
    proxy = shared.clone()
    shoulder_center = 0.5 * (shared[..., left_shoulder_index, :] + shared[..., right_shoulder_index, :])
    proxy[..., torso_index, :] = shared[..., 0, :] + float(alpha) * (shoulder_center - shared[..., 0, :])
    return proxy


def batched_pa_align(source: torch.Tensor, target: torch.Tensor, detach_transform: bool = False) -> torch.Tensor:
    if source.numel() == 0:
        return source
    # CUDA's batched SVD does not support FP16.  Tensor.float() alone is not
    # sufficient under AMP because autocast converts the surrounding matmuls
    # back to FP16 before the decomposition.
    with torch.autocast(device_type=source.device.type, enabled=False):
        source = source.float()
        target = target.float()
        mu_src = source.mean(dim=1, keepdim=True)
        mu_tgt = target.mean(dim=1, keepdim=True)
        src0 = source - mu_src
        tgt0 = target - mu_tgt
        var_src = torch.sum(src0 ** 2, dim=(1, 2)).clamp_min(1e-8)
        k_mat = torch.matmul(src0.transpose(1, 2), tgt0)
        u_mat, _, vh_mat = torch.linalg.svd(k_mat)
        v_mat = vh_mat.transpose(1, 2)
        rot = torch.matmul(v_mat, u_mat.transpose(1, 2))
        det = torch.linalg.det(rot)
        if torch.any(det < 0):
            v_mat = v_mat.clone()
            v_mat[det < 0, :, -1] *= -1.0
            rot = torch.matmul(v_mat, u_mat.transpose(1, 2))
        scale = torch.sum(torch.diagonal(torch.matmul(rot, k_mat), dim1=1, dim2=2), dim=1) / var_src
        if detach_transform:
            rot = rot.detach()
            scale = scale.detach()
            mu_src = mu_src.detach()
            mu_tgt = mu_tgt.detach()
        rotated_src = torch.matmul(source, rot.transpose(1, 2))
        rotated_mu_src = torch.matmul(mu_src, rot.transpose(1, 2))
        return scale[:, None, None] * rotated_src + (mu_tgt - scale[:, None, None] * rotated_mu_src)


def pa_align_sequence(pred_shared: torch.Tensor, target_shared: torch.Tensor, detach_transform: bool = False) -> torch.Tensor:
    leading_shape = pred_shared.shape[:-2]
    pred_flat = pred_shared.reshape(-1, pred_shared.shape[-2], 3)
    target_flat = target_shared.reshape(-1, target_shared.shape[-2], 3)
    aligned = batched_pa_align(pred_flat, target_flat, detach_transform=detach_transform)
    return aligned.reshape(*leading_shape, pred_shared.shape[-2], 3)


def gamp_pa_align_sequence(
    pred_shared: torch.Tensor,
    smpl_shared: torch.Tensor,
    edge_scale: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
    detach_transform: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_scaled = scale_shared_to_robot_shape(smpl_shared, edge_scale, parents=parents)
    pred_aligned = pa_align_sequence(pred_shared, target_scaled, detach_transform=detach_transform)
    return pred_aligned, target_scaled


def valid_shared(pred_shared: torch.Tensor, target_shared: torch.Tensor, valid: torch.Tensor):
    flat_valid = valid.reshape(-1) > 0.5
    pred_valid = pred_shared.reshape(-1, pred_shared.shape[-2], 3)[flat_valid]
    target_valid = target_shared.reshape(-1, target_shared.shape[-2], 3)[flat_valid]
    return pred_valid, target_valid
