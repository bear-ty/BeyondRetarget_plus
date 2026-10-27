"""Human/robot sole heights in one gravity-aligned frame; no floor snapping."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import model_dof_to_xml_dof
from lib.util.beyondretarget.static_ik import _foot_joint_specs, _marker_positions, _quat_xyzw_to_rotmat
from lib.util.beyondretarget.smpl_geometry import SMPLGeometry
from lib.utils.geometry import rot6d_to_rotmat_hmr2


class RobotFootGeometry:
    """Compile XML once, keep foot hulls in BODY coordinates (including geom
    position/rotation/scale applied by MuJoCo). FK transforms only those hulls.
    No renderer or GPU context. No silent ankle-marker fallback.
    """
    def __init__(self, spec, device='cpu'):
        import mujoco
        import yaml
        self.spec=spec;self.device=torch.device(device)
        self.kin=RobotKinematicsModel(spec.xml_path,self.device,model_to_xml_dof=None)
        paths=Path(__file__).resolve().parents[3]/'configs/static_geometry.yaml'
        options=yaml.safe_load(paths.read_text()).get(spec.name,{})
        if options.get('meshdir'):
            tree=ET.parse(spec.xml_path)
            compiler=tree.getroot().find('compiler')
            if compiler is None:raise ValueError('Explicit meshdir only supports MJCF robots')
            meshdir=Path(options['meshdir'])
            if not meshdir.is_absolute() and str(meshdir).startswith('assets/robot'):
                mesh_root = Path(os.environ.get('BRPLUS_ROBOT_ROOT', paths.parents[1] / 'assets' / 'robot'))
                meshdir = mesh_root / Path(str(meshdir)).relative_to('assets/robot')
            if not meshdir.is_dir():raise FileNotFoundError(meshdir)
            compiler.set('meshdir',str(meshdir))
            model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
        else:
            model=mujoco.MjModel.from_xml_path(spec.xml_path)
        self.parts=[];self.description=[]
        foot_specs=[next(x for x in spec.shared_joint_map if x.name==side+'_foot') for side in ('left','right')]
        for foot in foot_specs:
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,foot.robot_link)
            if body<1:raise ValueError('Foot body missing in XML: '+foot.robot_link)
            # Fixed semantic toe bodies often have no own geometry. Resolve
            # nearest foot/ankle ancestor; fail before climbing to a knee/hip.
            geom_ids=[]
            for _ in range(3):
                geom_ids=list(np.flatnonzero(model.geom_bodyid==body))
                if geom_ids:break
                body=int(model.body_parentid[body])
                if body==0:break
            if not geom_ids:raise ValueError('No physical foot geometry: '+foot.robot_link)
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body)
            body_id=self.kin.body_names.index(name)
            components=[]
            # Prefer actual visual meshes, not large proxy collision shapes.
            meshes=[g for g in geom_ids if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH]
            for g in meshes or geom_ids:
                typ=model.geom_type[g];size=model.geom_size[g]
                if typ==mujoco.mjtGeom.mjGEOM_MESH:
                    mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
                    vertices=model.mesh_vert[a:a+n].astype(np.float64)
                    vertices=vertices[ConvexHull(vertices).vertices]
                elif typ==mujoco.mjtGeom.mjGEOM_BOX:
                    vertices=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*size
                elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:
                    # An exact sphere minimum is represented separately.
                    components.append((torch.tensor(model.geom_pos[g],device=self.device,dtype=torch.float32)[None],float(size[0])))
                    continue
                else:
                    raise ValueError(f'Unsupported foot geometry type {typ}: {spec.name}/{name}; configure mesh explicitly')
                rotation=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix()
                vertices=vertices@rotation.T+model.geom_pos[g]
                components.append((torch.tensor(vertices,dtype=torch.float32,device=self.device),0.))
            self.parts.append((body_id,components))
            self.description.append(dict(semantic=foot.name,body=name,vertices=sum(len(v) for v,r in components)))
        self.foot_ids,self.offsets=_foot_joint_specs(spec,self.kin)

    @torch.no_grad()
    def evaluate(self, raw):
        root=torch.as_tensor(raw['root_trans'],dtype=torch.float32,device=self.device)
        quat=torch.as_tensor(raw['root_rot_quat'][:,[1,2,3,0]],dtype=torch.float32,device=self.device)
        dof=model_dof_to_xml_dof(torch.as_tensor(raw['dof'],dtype=torch.float32,device=self.device),self.spec)
        positions,quaternions=self.kin.forward_kinematics(root,quat,dof)
        rotations=_quat_xyzw_to_rotmat(quaternions)
        heights=[]
        for body,components in self.parts:
            zs=[positions[:,body,2,None]+rotations[:,body,2]@v.T-radius for v,radius in components]
            heights.append(torch.cat(zs,-1).amin(-1))
        markers=_marker_positions(positions,quaternions,self.foot_ids,
            torch.tensor(self.offsets,dtype=torch.float32,device=self.device))
        return torch.stack(heights,-1).cpu().numpy(),markers.cpu().numpy()


@torch.no_grad()
def predicted_human_sole_heights(pose6d, betas, trans_camera, camera_to_world_R, camera_to_world_T,
                                 device='cpu', batch_size=128):
    """Predictions only: HMR2-column pose, SMPL translation, world-Zup camera.

    Sparse LBS of 508 neutral-SMPL foot skin vertices is exactly the same as
    those vertices in a full SMPL forward. Foot extrema respect shape, pose,
    local rotation, and skinning; no per-frame full mesh/rendering.
    """
    n=len(pose6d);geom=SMPLGeometry().to(device).eval();result=[]
    for k in range(0,n,batch_size):
        def t(x):return torch.as_tensor(np.asarray(x[k:k+batch_size]),dtype=torch.float32,device=device)
        pose=rot6d_to_rotmat_hmr2(t(pose6d).reshape(-1,24,6)).reshape(-1,24,3,3)
        _,v=geom(pose,t(betas),pose_is_matrix=True)
        v=v+t(trans_camera).reshape(-1,1,3)
        world=torch.einsum('bij,bvj->bvi',t(camera_to_world_R),v)+t(camera_to_world_T)[:,None]
        result.append(torch.stack((world[:,:geom.left_count,2].amin(1),world[:,geom.left_count:,2].amin(1)),-1).cpu().numpy())
    return np.concatenate(result)


def align_root_height(raw, human_height, robot_height, static_prob, weight_floor=.25, clearance=0.):
    """Weighted least squares deltaZ; this step changes NOTHING except root Z.
    A single root shift cannot satisfy both feet when leg poses disagree.
    There is intentionally no hard switch, zero-floor rule, or hidden lift.
    """
    n=len(raw['root_trans']);arrays=(human_height,robot_height,static_prob)
    if n<1 or any(np.shape(x)!=(n,2) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Nonempty finite [T,2] heights/probabilities required')
    if weight_floor<=0 or np.any((static_prob<0)|(static_prob>1)) or not np.isfinite(clearance):
        raise ValueError('Invalid confidence weights / clearance')
    residual=human_height-robot_height
    if np.any(np.abs(residual)>3.):
        raise ValueError('Foot height discrepancy >3m: check world frame/units, not silently clip')
    weights=weight_floor+static_prob
    delta=(weights*residual).sum(1)/weights.sum(1)+clearance
    out={k:v.copy() for k,v in raw.items()};out['root_trans'][:,2]+=delta
    assert np.array_equal(out['root_trans'][:,:2],raw['root_trans'][:,:2])
    assert np.array_equal(out['root_rot_quat'],raw['root_rot_quat'])
    return out,delta.astype(np.float32)
