"""Small audited geometry helpers; 6D uses PyTorch3D ROW convention."""
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion


def masked_reduce_mean_g(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.bool().expand_as(values)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def rot6d_to_quat_wxyz(rot6d):
    with torch.autocast(device_type=rot6d.device.type, enabled=False):
        return matrix_to_quaternion(rotation_6d_to_matrix(rot6d.float()))


def rot6d_geodesic_g(pred, target, mask):
    relative = rotation_6d_to_matrix(pred.float()).transpose(-1, -2) @ rotation_6d_to_matrix(target.float())
    angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1+1e-6, 1-1e-6))
    return masked_reduce_mean_g(angle, mask)
