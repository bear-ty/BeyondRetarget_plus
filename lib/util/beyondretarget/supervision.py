"""Native TRAM SMPL mesh -> v5's neutral, vertex-regressed SMPL-24 reference."""
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from lib.models.smpl import SMPL


class SmplRobotReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.smpl = SMPL(gender='neutral')
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch):
        pose, betas = batch['pose'].float(), batch['betas'].float()
        b, t = pose.shape[:2]
        if pose.shape[-1] != 72 or betas.shape[-1] != 10:
            raise ValueError('Expected native TRAM SMPL pose [B,T,72], betas [B,T,10]')
        if not torch.isfinite(pose).all() or not torch.isfinite(betas).all():
            raise ValueError('Non-finite SMPL annotations')
        rot = axis_angle_to_matrix(pose.reshape(-1,24,3))
        with torch.autocast(device_type=pose.device.type, enabled=False):
            mesh = self.smpl(global_orient=rot[:,:1], body_pose=rot[:,1:],
                             betas=betas.reshape(-1,10), pose2rot=False, default_smpl=True)
            joints = torch.einsum('jv,bvc->bjc',self.smpl.J_regressor,mesh.vertices)
        # Translation cancels in every v5 loss/pose metric. BEDLAM 'trans' is NOT
        # verified camera translation: never report trajectory accuracy from it.
        target = {'smpl_joints':joints.reshape(b,t,24,3), 'root_pos':pose.new_zeros(b,t,3),
                  'root_rot_6d':matrix_to_rotation_6d(rot[:,0]).reshape(b,t,6),
                  'frame_mask':batch['frame_mask'].bool(), 'fps':batch['fps'],
                  'continuous_pair_mask':batch['continuous_pair_mask'].bool()}
        if not target['frame_mask'].any():
            raise ValueError('Batch has no valid SMPL supervision')
        return target
