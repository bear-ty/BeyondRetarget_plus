# Ported from RGB2robo GVRMR f016326ff27977008a5dc61198806417df57ea26.
"""V5: root-local semantic SE(3), GT dynamics, weak DoF smoothness.

No PA in the objective, no batch-dependent weighting, no ground/contact loss.
Root-localization uses known root parameters, never a fitted alignment.
All reductions exclude padding. Temporal derivatives use physical sample FPS.
"""
import math
import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix
from lib.util.beyondretarget.smpl_supervised_robot_loss import (
    SmplSupervisedRobotLoss, _root_localize_sequence, _velocity, _accel)
from lib.util.beyondretarget.gamp_pa_mpjpe_utils import pa_align_sequence


def masked_mean(x, mask):
    mask = mask.bool()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    return torch.where(mask, x, torch.zeros_like(x)).sum() / mask.sum().clamp_min(1)


def huber_norm(x):
    return F.smooth_l1_loss(torch.linalg.vector_norm(x, dim=-1),
                            torch.zeros_like(x[..., 0]), reduction='none')


class GvrmrV5Loss(SmplSupervisedRobotLoss):
    profile = 'root_local_se3_v5'
    term_names = ('position', 'rotation', 'velocity', 'acceleration', 'dof_smooth', 'limit')
    defaults = dict(position=1., rotation=.25, velocity=.1, acceleration=.025,
                    dof_smooth=.01, limit=1.)

    def __init__(self, *args, v5_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = dict(v5_config or {})
        if self.robot_to_target_coord != 'zup_to_yup':
            raise ValueError('V5 requires audited smpl_parameter_axes_v2 coordinates')
        self.weights_v5 = {k: float(self.config.get(k + '_weight', v))
                           for k, v in self.defaults.items()}
        self.temporal_warmup_epochs = int(self.config.get('temporal_warmup_epochs', 5))
        self.epoch_index = 0
        lo = self.kinematics.dof_lower_limits.clone().float()
        hi = self.kinematics.dof_upper_limits.clone().float()
        if self.robot_spec.model_to_xml_dof is not None:
            idx = torch.as_tensor(self.robot_spec.model_to_xml_dof, device=lo.device)
            lo, hi = lo[idx], hi[idx]
        if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (hi > lo).all():
            raise ValueError('V5 needs finite nonzero joint ranges')
        self.register_buffer('v5_lower', lo, persistent=False)
        self.register_buffer('v5_upper', hi, persistent=False)
        # Fixed robot scale, never fitted to each sample or prediction.
        with torch.no_grad():
            device = lo.device
            neutral_values = self.robot_spec.neutral_dof or [0.] * self.robot_spec.dof
            q = torch.tensor(neutral_values, device=device)
            if q.numel() != self.robot_spec.dof:
                if self.robot_spec.model_to_xml_dof is None:
                    raise ValueError('Neutral DoF width does not match robot model')
                q = q[torch.as_tensor(self.robot_spec.model_to_xml_dof, device=device)]
            q = q.reshape(1, 1, -1)
            neutral = {'dof': q, 'root_pos': torch.zeros(1, 1, 3, device=device),
                       'root_rot_6d': torch.tensor([1.,0,0,0,1,0], device=device).reshape(1,1,6)}
            points = self._shared_robot(neutral)[0, 0]
            legs = []
            for side in ('left', 'right'):
                ids = [self.shared_names.index(side + '_' + j) for j in ('hip','knee','ankle')]
                legs.append((points[ids[0]]-points[ids[1]]).norm() +
                            (points[ids[1]]-points[ids[2]]).norm())
            self.leg_length_m = float(torch.stack(legs).mean())
        if not .1 < self.leg_length_m < 3.:
            raise ValueError(f'Unexpected robot leg length {self.leg_length_m}')
        self.position_scale = self.leg_length_m * float(self.config.get('position_scale_leg', .05))
        self.velocity_scale = self.leg_length_m * float(self.config.get('velocity_scale_leg_s', .5))
        self.acceleration_scale = self.leg_length_m * float(self.config.get('acceleration_scale_leg_s2', 5.))
        self.rotation_scale_deg = float(self.config.get('rotation_scale_deg', 10.))
        self.dof_acc_scale = float(self.config.get('dof_acc_scale_s2', 5.))
        self.limit_scale = math.radians(float(self.config.get('limit_scale_deg', 3.)))

    def set_epoch(self, epoch):
        self.epoch_index = int(epoch)

    def _frames(self, p):
        # Geometric branch always FP32, including cross-product normalization.
        zraw = p[..., self.se3_primary_indices, :] - p
        ref = p[..., self.se3_reference_indices, :] - p
        eps = 1e-4 * self.leg_length_m
        z = zraw / zraw.norm(dim=-1, keepdim=True).clamp_min(eps)
        ref_unit = ref / ref.norm(dim=-1, keepdim=True).clamp_min(eps)
        yraw = torch.cross(z, ref_unit, dim=-1)
        sin_angle = yraw.norm(dim=-1)
        y = yraw / sin_angle.unsqueeze(-1).clamp_min(1e-4)
        xraw = torch.cross(y, z, dim=-1)
        x = xraw / xraw.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        valid_axis = zraw.norm(dim=-1) > eps
        valid_frame = valid_axis & (ref.norm(dim=-1) > eps) & (sin_angle > .1)
        return torch.stack((x,y,z), dim=-1), z, valid_axis, valid_frame

    def _geometry(self, pred, target):
        p = self._shared_robot(pred).float()
        g = self._target_shared_gamp(self._anchor_shared_root(
            self._shared_smpl(target), target['root_pos'])).float()
        plocal = _root_localize_sequence(p, pred['root_rot_6d'].float())
        glocal = _root_localize_sequence(g, target['root_rot_6d'].float())
        return p, g, plocal, glocal

    def _calculate(self, pred, target, eval_metrics=False):
        valid = target['frame_mask'] > .5
        p, g, plocal, glocal = self._geometry(pred, target)
        # Pelvis is identically zero in the root-local loss, exclude it.
        point_mask = valid[..., None].expand(*valid.shape, plocal.shape[-2]).clone()
        point_mask[..., 0] = False
        pf, paxis, _, pfvalid = self._frames(plocal)
        gf, gaxis, ga_valid, gfvalid = self._frames(glocal)
        rot_norm = 8 * math.sin(math.radians(self.rotation_scale_deg) / 2)**2
        rot_full = (pf - gf).square().sum(dim=(-1,-2)) / rot_norm
        rot_axis = (paxis-gaxis).square().sum(-1) / (4*math.sin(math.radians(self.rotation_scale_deg)/2)**2)
        # Target alone decides observability: predicted degeneracy cannot hide errors.
        rot_value = torch.where(gfvalid, rot_full, rot_axis)
        rot_mask = point_mask & ga_valid
        raw = {'position': masked_mean(huber_norm((plocal-glocal)/self.position_scale), point_mask),
               'rotation': masked_mean(rot_value, rot_mask)}
        fps = target.get('fps', self.fps)
        pair = valid[:, 1:] & valid[:, :-1]
        triple = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        if 'continuous_pair_mask' in target:
            pair = pair & target['continuous_pair_mask'].bool()
            triple = triple & pair[:, 1:] & pair[:, :-1]
        vel = _velocity(plocal, fps)-_velocity(glocal, fps)
        acc = _accel(plocal, fps)-_accel(glocal, fps)
        raw['velocity'] = masked_mean(huber_norm(vel[...,1:,:]/self.velocity_scale), pair)
        raw['acceleration'] = masked_mean(huber_norm(acc[...,1:,:]/self.acceleration_scale), triple)
        span = self.v5_upper-self.v5_lower
        qnorm = (pred['dof'].float()-(self.v5_upper+self.v5_lower)/2)/span
        qacc = _accel(qnorm, fps)
        raw['dof_smooth'] = masked_mean(F.smooth_l1_loss(qacc/self.dof_acc_scale,
                                                  torch.zeros_like(qacc), reduction='none'), triple)
        overflow = F.relu(self.v5_lower-pred['dof'].float()) + F.relu(pred['dof'].float()-self.v5_upper)
        raw['limit'] = masked_mean((overflow/self.limit_scale).square(), valid)
        ramp = min(1., (self.epoch_index+1)/max(self.temporal_warmup_epochs, 1)) if self.training else 1.
        weighted = {k: v*self.weights_v5[k]*(ramp if k in ('velocity','acceleration','dof_smooth') else 1.)
                    for k,v in raw.items()}
        total = sum(weighted.values())
        if not torch.isfinite(total):
            raise RuntimeError('Non-finite v5 loss')
        stats = {'loss': total, 'v5_temporal_ramp': total.new_tensor(ramp)}
        for k in raw:
            stats['loss_v5_'+k] = raw[k]
            stats['loss_v5_'+k+'_weighted'] = weighted[k]
        stats['v5_full_strength_loss'] = sum(raw[k]*self.weights_v5[k] for k in raw)
        stats['v5_selection_score'] = raw['position'] + self.weights_v5['rotation']*raw['rotation']
        stats['v5_local_mpjpe_mm'] = masked_mean((plocal-glocal).norm(dim=-1)*1000, point_mask)
        cosine = ((pf.transpose(-1,-2) @ gf).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        angle = torch.rad2deg(torch.acos(cosine.detach().clamp(-1,1)))
        stats['v5_rotation_deg'] = masked_mean(angle, point_mask & gfvalid & pfvalid)
        stats['v5_frame_valid_rate'] = masked_mean(gfvalid.float(), point_mask)
        stats['v5_pred_frame_valid_rate'] = masked_mean(pfvalid.float(), point_mask)
        stats['v5_velocity_error_m_s'] = masked_mean(vel[...,1:,:].norm(dim=-1), pair)
        stats['v5_acceleration_error_m_s2'] = masked_mean(acc[...,1:,:].norm(dim=-1), triple)
        stats['v5_dof_acc_rms_rad_s2'] = masked_mean(_accel(pred['dof'].float(),fps).square(), triple).sqrt()
        stats['v5_limit_over_rate'] = masked_mean((overflow>0).float(), valid)
        for label,pairs in [('torso',self.torso_dir_pairs),('thigh',self.thigh_dir_pairs),('shank',self.shank_dir_pairs),('arm',self.arm_segment_dir_pairs)]:
            stats['v5_'+label+'_deg'] = self._direction_error_deg(plocal, glocal, valid, pairs).detach()
        if eval_metrics:
            rootrel_p = p-p[...,:1,:]
            rootrel_g = g-g[...,:1,:]
            stats['v5_incam_root_relative_mm'] = masked_mean((rootrel_p-rootrel_g).norm(dim=-1)*1000, point_mask)
            pa = pa_align_sequence(p, g)
            stats['smpl_gamp_pa_mpjpe_mm'] = masked_mean((pa-g).norm(dim=-1)*1000, point_mask)
            rel = rotation_6d_to_matrix(pred['root_rot_6d'].float()).transpose(-1,-2) @ rotation_6d_to_matrix(target['root_rot_6d'].float())
            rootangle = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
            stats['v5_frozen_root_rotation_deg'] = masked_mean(rootangle,valid)
            stats['v5_frozen_root_position_mm'] = masked_mean((pred['root_pos']-target['root_pos']).norm(dim=-1)*1000,valid)
        return stats

    def forward(self, pred, target):
        with torch.autocast(device_type=pred['dof'].device.type, enabled=False):
            return self._calculate(pred, target)

    @torch.no_grad()
    def compute_metrics(self, pred, target):
        with torch.autocast(device_type=pred['dof'].device.type, enabled=False):
            return self._calculate(pred, target, eval_metrics=True)
