"""Exact SMPL skeleton and sparse foot LBS; CPU-safe and mesh-free for labels."""
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.lbs import batch_rigid_transform
from lib.models.smpl import SMPL


class SMPLGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        model = SMPL()
        reg = model.J_regressor
        self.register_buffer('rest_joints', reg @ model.v_template)
        self.register_buffer('shape_joints', torch.einsum('jv,vck->jck', reg, model.shapedirs))
        self.register_buffer('parents', model.parents)
        # Keep all skin vertices assigned mainly to an ankle/foot. Their sparse
        # LBS uses all 24 joints, shape and pose blend shapes, not rigid offsets.
        left = (model.lbs_weights[:, [7, 10]].sum(-1) > .5).nonzero().flatten()
        right = (model.lbs_weights[:, [8, 11]].sum(-1) > .5).nonzero().flatten()
        ids = torch.cat((left, right))
        self.left_count = len(left)
        self.register_buffer('vertex_ids', ids)
        self.register_buffer('template', model.v_template[ids])
        self.register_buffer('shapedirs', model.shapedirs[ids])
        self.register_buffer('posedirs', model.posedirs.reshape(207, -1, 3)[:, ids].reshape(207, -1))
        self.register_buffer('weights', model.lbs_weights[ids])

    def skeleton(self, pose, betas, pose_is_matrix=False):
        rotation = pose if pose_is_matrix else axis_angle_to_matrix(pose.reshape(-1, 24, 3))
        rest = self.rest_joints[None] + torch.einsum('bk,jck->bjc', betas, self.shape_joints)
        joints, transforms = batch_rigid_transform(rotation, rest, self.parents)
        return joints, transforms, rotation

    def forward(self, pose, betas, pose_is_matrix=False):
        joints, transforms, rotation = self.skeleton(pose, betas, pose_is_matrix)
        shape = self.template[None] + torch.einsum('bk,vck->bvc', betas, self.shapedirs)
        identity = torch.eye(3, dtype=pose.dtype, device=pose.device)
        feature = (rotation[:, 1:] - identity).reshape(len(pose), 207)
        posed = shape + (feature @ self.posedirs).reshape(len(pose), -1, 3)
        blended = torch.einsum('vj,bjkl->bvkl', self.weights, transforms)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
        vertices = torch.einsum('bvij,bvj->bvi', blended, homogeneous)[..., :3]
        return joints, vertices
