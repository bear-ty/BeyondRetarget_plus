"""Approved offline C-RC1: observation-guided translation + image-supported XY/IK.

Inputs must describe ONE continuous shot in world Z-up meters, real FPS,
model-order DoFs, wxyz root quaternions. No clip IDs or action classes.
Reprojection and the approved v2 pose filter precede this module. Cached
pre-filter results are accepted for the present reproducibility audit.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import binary_opening
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import model_dof_to_xml_dof
from lib.util.beyondretarget.motion_filter_v2 import difference, smooth_euclidean
from lib.util.beyondretarget.static_ik import (
    _foot_joint_specs, _marker_positions, _ccd_solve_sequence,
    _smooth_ik_delta, FootIKConfig,
)


@dataclass(frozen=True)
class CRC1Config:
    probability_min: float = .5
    image_speed_max_bodyheight_s: float = .08
    image_spread_max_bodyheight_s: float = .16
    minimum_support_seconds: float = .1
    constraint_weight_min: float = .95
    fade_seconds: float = .1
    root_position_sigma_m: float = .25
    root_acceleration_sigma_mps2: float = 3.
    support_speed_sigma_mps: float = .05
    ik_iterations: int = 2
    ik_step_scale: float = .1
    ik_cap_degrees: float = 5.

    def __post_init__(self):
        if not 0 <= self.probability_min < 1 or not .8 < self.constraint_weight_min <= 1:
            raise ValueError('Invalid C-RC1 probability/constraint thresholds')
        for name in ('image_speed_max_bodyheight_s', 'image_spread_max_bodyheight_s',
                     'minimum_support_seconds', 'fade_seconds', 'root_position_sigma_m',
                     'root_acceleration_sigma_mps2', 'support_speed_sigma_mps',
                     'ik_step_scale', 'ik_cap_degrees'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Expected positive finite ' + name)
        if not isinstance(self.ik_iterations, int) or self.ik_iterations < 1:
            raise ValueError('Expected positive integer IK iterations')


def validate_motion(raw, filtered, fitted_world_root, fps):
    ids=np.asarray(raw['frame']);n=len(ids)
    if n<1 or not np.isfinite(fps) or fps<=0:
        raise ValueError('Nonempty clip and positive finite FPS required')
    if ids.ndim!=1 or not np.equal(ids,np.floor(ids)).all() or np.any(np.diff(ids)!=1):
        raise ValueError('Split temporal gaps/shots before this offline solver')
    for key in ['root_trans','root_rot_quat','dof']:
        a=np.asarray(filtered[key])
        if len(a)!=n or not np.isfinite(a).all():raise ValueError('Invalid '+key)
    r=np.asarray(fitted_world_root);p=np.asarray(raw['static_prob6'])
    if r.shape!=(n,3) or not np.isfinite(r).all():raise ValueError('Invalid fitted root')
    if p.shape!=(n,6) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected probabilities, not support weights')
    if not np.allclose(np.linalg.norm(filtered['root_rot_quat'],axis=1),1,atol=1e-4):
        raise ValueError('Invalid unit quaternion')
    if not np.isfinite(raw['root_pos_world_zup']).all():raise ValueError('Invalid raw root')


def image_support_weights(probability, evidence, fps, cfg=CRC1Config()):
    p=np.asarray(probability);n=len(p)
    if p.shape!=(n,6) or n<1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected finite [T,6] probabilities')
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
    speed=np.asarray(evidence['speed_bodyheight_s'])
    spread=np.asarray(evidence['uncertainty_bodyheight_s'])
    good=np.asarray(evidence['evidence'],bool)
    if speed.shape!=(n,2) or spread.shape!=(n,2) or good.shape!=(n,2):
        raise ValueError('Image evidence/frame mismatch')
    accepted=(good & np.isfinite(speed) & np.isfinite(spread)
              & (speed<cfg.image_speed_max_bodyheight_s)
              & (spread<cfg.image_spread_max_bodyheight_s)
              & (p[:,[1,3]]>cfg.probability_min))
    accepted[-1]=False  # Last row has no outgoing interval.
    duration=max(1,int(round(cfg.minimum_support_seconds*fps)))
    for j in range(2):
        # Opening removes short episodes; it does NOT fill missing evidence.
        accepted[:,j]=binary_opening(accepted[:,j],structure=np.ones(duration))
    weights=np.zeros_like(p)
    weights[:,[1,3]]=np.where(accepted,np.maximum(p[:,[1,3]],cfg.constraint_weight_min),0.)
    return weights


def joint_xy_solve(prior, relative_feet, support_weights, fps, cfg=CRC1Config()):
    r=np.asarray(prior,dtype=np.float64);n=len(r)
    if r.shape!=(n,3) or np.shape(relative_feet)!=(n,2,3) or np.shape(support_weights)!=(n,2):
        raise ValueError('Joint solve input shape mismatch')
    if not all(np.isfinite(x).all() for x in [r,relative_feet,support_weights]) or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid joint solve values')
    if n<2:return r.copy()
    d1=difference(n,1);d2=difference(n,2)
    active=(np.asarray(support_weights)[:-1]>.8).astype(float)
    fade=max(1,int(round(cfg.fade_seconds*fps)))
    for j in range(2):
        ids=np.flatnonzero(active[:,j])
        for span in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if len(span):
                active[span,j]*=np.minimum(1.,np.minimum(np.arange(1,len(span)+1),np.arange(len(span),0,-1))/fade)
    target=-(np.diff(relative_feet[:,:,:2],axis=0)*active[:,:,None]).sum(1)
    w=active.sum(1)
    beta=fps**2/cfg.support_speed_sigma_mps**2
    alpha=fps**4/cfg.root_acceleration_sigma_mps2**2
    a=eye(n,format='csc')/cfg.root_position_sigma_m**2+alpha*(d2.T@d2)+beta*d1.T@diags(w)@d1
    rhs=r[:,:2]/cfg.root_position_sigma_m**2+beta*d1.T@target
    result=r.copy();result[:,:2]=spsolve(a.tocsc(),rhs)
    return result


def foot_positions(motion, spec, kin=None):
    kin=kin or RobotKinematicsModel(spec.xml_path,'cpu')
    ids,offsets=_foot_joint_specs(spec,kin)
    root=torch.as_tensor(motion['root_trans'],dtype=torch.float32)
    quat=torch.as_tensor(motion['root_rot_quat'][:,[1,2,3,0]],dtype=torch.float32)
    dof=model_dof_to_xml_dof(torch.as_tensor(motion['dof'],dtype=torch.float32),spec)
    offsets=torch.tensor(offsets,dtype=torch.float32)
    with torch.no_grad():
        pos,rot=kin.forward_kinematics(root,quat,dof)
        feet=_marker_positions(pos,rot,ids,offsets).numpy()
    return feet,kin,ids,offsets,quat,dof


def bounded_foot_ik(motion, spec, support, fps, cfg=CRC1Config()):
    # Exact production CCD/target/delta-smoothing semantics; no monkeypatches.
    feet,kin,ids,offsets,quat,dof=foot_positions(motion,spec)
    # Production balanced-v2 promotes the post-trajectory foot positions to
    # float64 before recursive targets; retain that order for exact replay.
    feet=feet.astype(np.float64)
    n=len(feet);targets=feet.copy()
    for t in range(1,n):
        w=support[t-1,:,None]
        targets[t]=w*targets[t-1]+(1-w)*feet[t]
    with torch.no_grad():
        refined=_ccd_solve_sequence(kin,torch.as_tensor(motion['root_trans'],dtype=torch.float32),quat,dof,
            torch.as_tensor(targets,dtype=torch.float32),ids,offsets,torch.ones((n,2),dtype=torch.bool),
            max_iter=cfg.ik_iterations,step_scale=cfg.ik_step_scale,enable_contact_foot_flat=False)
    refined=_smooth_ik_delta(dof.numpy(),refined.numpy(),kin.dof_lower_limits.numpy(),kin.dof_upper_limits.numpy(),
        FootIKConfig(fps=float(fps),enable_contact_foot_flat=False,
                     ik_delta_smooth_window=max(1,round(5*fps/30)),ik_delta_rate_limit=.18*30/fps))
    refined=refined if spec.model_to_xml_dof is None else refined[:,spec.model_to_xml_dof]
    lower=kin.dof_lower_limits.numpy();upper=kin.dof_upper_limits.numpy()
    if spec.model_to_xml_dof is not None:lower=lower[spec.model_to_xml_dof];upper=upper[spec.model_to_xml_dof]
    cap=np.deg2rad(cfg.ik_cap_degrees)
    q=motion['dof']+cap*np.tanh((refined-motion['dof'])/cap)
    out={k:v.copy() for k,v in motion.items()}
    out['dof']=np.clip(q,lower,upper).astype(np.float32)
    return out


def refine_from_prefilter(raw, filtered, fitted_world_root, evidence, spec, fps,
                          cfg=CRC1Config()):
    validate_motion(raw,filtered,fitted_world_root,fps)
    pre={k:np.asarray(v).copy() for k,v in filtered.items()}
    delta=np.asarray(fitted_world_root)-raw['root_pos_world_zup']
    # v2 translation smoother is linear: S(root+delta)=S(root)+S(delta).
    # Existing robot/human sole-alignment offset is therefore retained.
    pre['root_trans']=pre['root_trans']+smooth_euclidean(delta,64,fps)
    support=image_support_weights(raw['static_prob6'],evidence,fps,cfg)
    feet=foot_positions(pre,spec)[0]
    solved={k:v.copy() for k,v in pre.items()}
    solved['root_trans']=joint_xy_solve(pre['root_trans'],feet-pre['root_trans'][:,None],support[:,[1,3]],fps,cfg)
    final=bounded_foot_ik(solved,spec,support[:,[1,3]],fps,cfg)
    assert np.array_equal(final['root_trans'][:,2],pre['root_trans'][:,2])
    assert np.array_equal(final['root_rot_quat'],filtered['root_rot_quat'])
    return final,support,{'prefilter':pre,'joint_xy':solved},delta
