"""Approved offline balanced filter v2 (G1, four-video audit 2026-10-17).

SO(3) angular-acceleration smoothing, XYZ third differences, and bounded
joint/FK third differences. No action recognition, spin counting or GT input.
Do not use this full-sequence optimization as a streaming latency claim.
"""
import logging
import time
import numpy as np
import torch
from scipy.optimize import least_squares, lsq_linear, minimize
from scipy.spatial.transform import Rotation
from scipy.sparse import eye, lil_matrix, vstack, diags
from scipy.sparse.linalg import spsolve
from lib.robot.robot_spec import model_dof_to_xml_dof
from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.util.beyondretarget.static_filter import _quat_wxyz_to_rot6d_mte

LOG=logging.getLogger(__name__)


def difference(n,order):
    d=eye(n,format='csr')
    for _ in range(order):d=d[1:]-d[:-1]
    return d


def smooth_euclidean(values,weight=64.,fps=30.):
    y=np.asarray(values,dtype=np.float64)
    if len(y)<4:return y.copy()
    d=difference(len(y),3)
    a=eye(len(y),format='csc')+weight*(fps/30.)**6*(d.T@d)
    return spsolve(a.tocsc(),y).reshape(y.shape)


def smooth_so3(quat,weight=4.,cap_deg=20.,fps=30.,max_nfev=40):
    raw=Rotation.from_quat(np.asarray(quat)[:,[1,2,3,0]])
    n=len(quat)
    if n<4:return raw.as_quat()[:,[3,0,1,2]],dict(success=True,nfev=0)
    step=(raw[1:]*raw[:-1].inv()).magnitude()
    adjacent=np.maximum(np.r_[step[0],step],np.r_[step,step[-1]])
    limit=np.minimum(np.deg2rad(cap_deg),.24*np.maximum(np.pi-adjacent,1e-5))
    pattern=lil_matrix((3*n+3*(n-2),3*n),dtype=int)
    for i in range(n):pattern[3*i:3*i+3,3*i:3*i+3]=1
    for i in range(n-2):pattern[3*n+3*i:3*n+3*i+3,3*i:3*(i+3)]=1
    def build(x):
        x=x.reshape(n,3);length=np.linalg.norm(x,axis=1);scale=np.ones(n);valid=length>1e-12
        scale[valid]=limit[valid]*np.tanh(length[valid]/limit[valid])/length[valid]
        delta=x*scale[:,None]
        return Rotation.from_rotvec(delta)*raw,delta
    def residual(x):
        r,delta=build(x)
        velocity=(r[1:]*r[:-1].inv()).as_rotvec()
        return np.r_[delta.ravel(),np.sqrt(weight)*(fps/30.)**2*np.diff(velocity,axis=0).ravel()]
    result=least_squares(residual,np.zeros(n*3),jac_sparsity=pattern.tocsr(),method='trf',
        ftol=2e-6,xtol=2e-6,gtol=2e-6,max_nfev=max_nfev)
    if not result.success:raise RuntimeError('SO3 filter did not converge: '+str(result.message))
    r,delta=build(result.x);q=r.as_quat()[:,[3,0,1,2]]
    for i in range(1,n):
        if q[i]@q[i-1]<0:q[i]*=-1
    return q,dict(success=True,nfev=result.nfev,
        maximum_raw_deviation_deg=float(np.rad2deg(np.linalg.norm(delta,axis=1)).max()))


@torch.inference_mode(False)
@torch.enable_grad()
def smooth_joints_fk(values,spec,fps=30.,joint_weight=4.,fk_weight=2.,length_scale=.3):
    """min ||q-raw||² + 4||D3 q||² + 2||D3 FK_local(q)/.3m||², in bounds."""
    kin=RobotKinematicsModel(spec.xml_path,'cpu')
    mapping=spec.model_to_xml_dof
    lower=kin.dof_lower_limits.numpy();upper=kin.dof_upper_limits.numpy()
    if mapping is not None:lower=lower[mapping];upper=upper[mapping]
    lower=lower.astype(float);upper=upper.astype(float)
    y=np.asarray(values,dtype=np.float64);n=len(y)
    if n<4:return np.clip(y,lower,upper),dict(success=True,iterations=0)
    lam=joint_weight*(fps/30.)**6;flam=fk_weight*(fps/30.)**6
    initial=smooth_euclidean(y,joint_weight,fps)
    d=difference(n,3);a=vstack([eye(n),np.sqrt(lam)*d],format='csr')
    for j in range(y.shape[1]):
        if initial[:,j].min()>=lower[j] and initial[:,j].max()<=upper[j]:continue
        result=lsq_linear(a,np.r_[y[:,j],np.zeros(n-3)],bounds=(lower[j],upper[j]),
            tol=1e-9,lsmr_tol=1e-11,max_iter=200)
        if not result.success:raise RuntimeError('Bounded filter initialization failed')
        initial[:,j]=result.x
    marker_ids=[kin.body_names.index(x.robot_link) for x in spec.shared_joint_map if x.name!='pelvis']
    if not marker_ids:raise ValueError('No semantic markers for FK smoothing')
    target=torch.tensor(y,dtype=torch.float64)
    zero=torch.zeros(n,3,dtype=torch.float64);identity=torch.zeros(n,4,dtype=torch.float64);identity[:,3]=1
    def objective(x):
        q=torch.tensor(x.reshape(y.shape),dtype=torch.float64,requires_grad=True)
        pos=kin.forward_kinematics(zero,identity,model_dof_to_xml_dof(q,spec))[0][:,marker_ids]/length_scale
        loss=(q-target).square().sum()+lam*torch.diff(q,n=3,dim=0).square().sum()
        loss=loss+flam*torch.diff(pos,n=3,dim=0).square().sum()
        grad=torch.autograd.grad(loss,q)[0]
        return float(loss.detach()),grad.detach().numpy().ravel()
    # The model inference caller may be in no_grad; this is pose optimization,
    # not model training. No network parameter is referenced by this objective.
    with torch.inference_mode(False),torch.enable_grad():
        result=minimize(objective,initial.ravel(),method='L-BFGS-B',jac=True,
            bounds=list(zip(np.tile(lower,n),np.tile(upper,n))),
            options=dict(maxiter=1500,ftol=1e-9,gtol=1e-5,maxcor=20))
    if not result.success:raise RuntimeError('Joint/FK filter did not converge: '+str(result.message))
    return result.x.reshape(y.shape),dict(success=True,iterations=result.nit,objective=float(result.fun))


def balanced_motion_filter(raw,spec,fps,*,so3_max_nfev=40):
    start=time.monotonic()
    quat,rotation_info=smooth_so3(raw['root_rot_quat'],fps=fps,max_nfev=so3_max_nfev)
    dof,joint_info=smooth_joints_fk(raw['dof'],spec,fps=fps)
    out=dict(fps=np.asarray(fps,dtype=np.float64),root_trans=smooth_euclidean(raw['root_trans'],64,fps),
        root_rot_quat=quat,root_rot_6d=_quat_wxyz_to_rot6d_mte(quat),dof=dof)
    LOG.info('FILTER balanced_v2 | SO3=%s | joint/FK=%s | %.2fs',rotation_info,joint_info,time.monotonic()-start)
    return out


def smooth_static_xy(root,relative_feet,probability,fps=30.,weight=16.,static_weight=256.):
    """Smooth corrected XY with the SAME already-validated static constraints."""
    root=np.asarray(root,float);n=len(root)
    if n<4:return root.copy()
    d1=difference(n,1);d3=difference(n,3)
    active=(probability[:-1]>.8).astype(float);weights=active.sum(1)
    target=-(np.diff(relative_feet[:,:,:2],axis=0)*active[:,:,None]).sum(1)
    beta=static_weight*(fps/30.)**2
    mat=eye(n,format='csc')+weight*(fps/30.)**6*(d3.T@d3)+beta*d1.T@diags(weights)@d1
    out=root.copy();out[:,:2]=spsolve(mat.tocsc(),root[:,:2]+beta*(d1.T@target))
    return out
