"""Approved C-RC1 translation fit. Fixed pose/shape/camera; no action rules."""
import numpy as np
import torch
from scipy.sparse import block_diag,kron,eye,diags
from scipy.sparse.linalg import spsolve
from lib.util.beyondretarget.smpl_geometry import SMPLGeometry
from lib.utils.geometry import rot6d_to_rotmat_hmr2
from lib.util.beyondretarget.motion_filter_v2 import difference

COCO=np.array([5,6,7,8,9,10,11,12,13,14,15,16])
SMPL=np.array([16,17,18,19,20,21,1,2,4,5,7,8])
def body_joints(raw):
    g=SMPLGeometry().eval()
    with torch.no_grad():
        r=rot6d_to_rotmat_hmr2(torch.tensor(raw['pred_pose']).reshape(-1,24,6)).reshape(-1,24,3,3)
        j=g.skeleton(r,torch.tensor(raw['pred_shape']),pose_is_matrix=True)[0].numpy()
    return j
def fit_translation(raw,kp,bbox,accel_xy=3.,sigma_px=3.,accel_z=15.,iterations=20):
    """Only root translation; fixed native SMPL pose/shape, no robot, static or teacher input.

    Robust 2D joint reprojection + world translation acceleration prior + weak
    original-root position prior. Metric camera extrinsics/intrinsics unchanged.
    """
    n=len(kp);fps=float(raw['fps'])
    if np.shape(kp)!=(n,17,3) or not np.isfinite(kp).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Expected finite row-aligned COCO17 keypoints and FPS')
    if len(raw['frame'])!=n or np.any(np.diff(raw['frame'])!=1):
        raise ValueError('Split gaps/shots; do not interpolate tracks in the root solver')
    if np.mean((kp[:,COCO,2]>.4).sum(1)>=4)<.8:
        raise ValueError('Insufficient reliable 2D observations; do not force this optimizer')
    j=body_joints(raw)[:,SMPL];obs=kp[:,COCO,:2].astype(float)
    conf=kp[:,COCO,2].astype(float);conf=np.where(conf>.4,conf,0.)
    rot=raw['camera_to_world_zup_R'].astype(float);trans=raw['camera_to_world_zup_T'].astype(float)
    init=raw['root_pos_world_zup'].astype(float);world=init.copy()
    f=np.broadcast_to(bbox['img_focal'],(n,)).astype(float);c=np.broadcast_to(bbox['img_center'],(n,2)).astype(float)
    if (n < 1 or not all(np.isfinite(x).all() for x in (j, rot, trans, init, f, c))
            or np.any(f <= 0) or min(accel_xy, accel_z, sigma_px) <= 0):
        raise ValueError('Invalid translation fit geometry/intrinsics/prior scales')
    D=kron(difference(n,2),np.diag([fps**2/accel_xy,fps**2/accel_xy,fps**2/accel_z]),format='csc')
    regular=D.T@D+eye(n*3,format='csc')*.01
    def project(x):
        root=np.einsum('nji,nj->ni',rot,x-trans);xyz=j+root[:,None]
        return xyz[:,:,:2]/xyz[:,:,2:]*f[:,None,None]+c[:,None],xyz
    initial_xyz=j+np.einsum('nji,nj->ni',rot,init-trans)[:,None]
    if initial_xyz[:,:,2].min() < .2:
        raise ValueError('Initial human joints behind/too close to camera; cannot safely fit translation')
    history=[]
    for it in range(iterations):
        pred,xyz=project(world);res=pred-obs
        # Huber image residual protects against swapped/occluded keypoints.
        mag=np.linalg.norm(res,axis=-1);weight=conf**2*np.minimum(1.,8./np.maximum(mag,1e-8))/sigma_px**2
        jac=np.zeros((n,len(SMPL),2,3));z=xyz[:,:,2]
        jac[:,:,0,0]=f[:,None]/z;jac[:,:,1,1]=f[:,None]/z
        jac[:,:,0,2]=-f[:,None]*xyz[:,:,0]/z**2;jac[:,:,1,2]=-f[:,None]*xyz[:,:,1]/z**2
        jac=jac@rot[:,None].transpose(0,1,3,2)
        H=np.einsum('njki,njkl,nj->nil',jac,jac,weight)
        grad=np.einsum('njki,njk,nj->ni',jac,res,weight)
        lhs=block_diag(list(H),format='csc')+regular
        rhs=-grad.ravel()-D.T@(D@world.ravel())-.01*(world-init).ravel()
        step=spsolve(lhs,rhs).reshape(n,3)
        if not np.isfinite(step).all():raise RuntimeError('Non-finite translation optimizer step')
        def objective(x):
            pp,xx=project(x)
            if xx[:,:,2].min()<.2:return np.inf
            rr=np.linalg.norm(pp-obs,axis=-1);robust=np.where(rr<=8,.5*rr**2,8*rr-32)
            return float(np.sum(robust*conf**2)/sigma_px**2+.5*np.square(D@x.ravel()).sum()+.005*np.square(x-init).sum())
        before=objective(world);scale=1.
        while scale>1e-4 and objective(world+step*scale)>before:scale*=.5
        after=objective(world+step*scale)
        if not np.isfinite(after) or after>before:
            # Keep the last valid iterate, never publish a failed line-search step.
            history.append({'iteration':it,'objective':before,'step':0.,'scale':0.,'line_search_stopped':True})
            break
        world+=step*scale;history.append({'iteration':it,'objective':objective(world),'step':float(np.abs(step*scale).max()),'scale':scale})
        if np.abs(step*scale).max()<1e-5:break
    initial_pred,_=project(init);final_pred,_=project(world)
    valid=conf>.4
    report={'accel_xy_prior_m_s2':accel_xy,'accel_z_prior_m_s2':accel_z,'sigma_px':sigma_px,'history':history,
        'raw_reprojection_median_px':float(np.median(np.linalg.norm(initial_pred-obs,axis=-1)[valid])),
        'fit_reprojection_median_px':float(np.median(np.linalg.norm(final_pred-obs,axis=-1)[valid])),
        'max_root_change_m':float(np.linalg.norm(world-init,axis=-1).max()),'uses_GVRMR_prediction':False,
        'note':'2D fit and smoothness are optimized criteria, not independent ground-truth validation.'}
    return world,report
