"""Reconstruct true BEDLAM cameras like GVHMR; preserve original labels/cache.

TRAM's cam_ext is NOT the physical world->camera transform for orbit clips.
Use official matching world/incam root parameters, including shaped-pelvis
offset, to reconstruct it. New labels use native TRAM SMPL joints in that
true world frame. No new images/features or inference GT input are introduced.
"""
import argparse,json,os,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from lib.datasets.robot_video_dataset import MappedNPZ,frame_number
from lib.models.tram_static import STATIC_VERSION,JOINT_IDS,JOINT_NAMES
from lib.util.beyondretarget.static_labels import annotation_identity,labels_from_positions,verify_label_manifest
from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.smpl_geometry import SMPLGeometry

POLICY='native_smpl_true_world_from_gvhmr_v2'


def reconstruct_camera(d):
    rw=Rotation.from_rotvec(np.asarray(d['pose'][:,:3])).as_matrix()
    rc=Rotation.from_rotvec(np.asarray(d['global_orient_incam'])).as_matrix()
    r=rc@rw.transpose(0,2,1)
    tw=np.asarray(d['trans'],np.float64)
    tc=np.asarray(d['trans_incam'],np.float64)+np.asarray(d['cam_ext'][:,:3,3],np.float64)
    offset=np.asarray(d['skeleton'][0],np.float64)
    t=tc+offset-np.einsum('nij,nj->ni',r,tw+offset)
    return r,t


def save_json(path,record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/static_labels/world_static6_truecamera_v2')
    p.add_argument('--motion-file',required=True,help='SMPL motion labels produced by the dataset preprocessing pipeline')
    p.add_argument('--annotation',default='data/datasets/dataset_ann/bedlam_vid.npz')
    p.add_argument('--old-labels',default='outputs/static_labels/world_static6_v1')
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4)
    root=Path(args.output);folder=root/'bedlam_vid';folder.mkdir(parents=True,exist_ok=True)
    db=MappedNPZ(args.annotation);n=len(db['imgname']);begin=time.monotonic()
    camera_manifest=folder/'camera_manifest.json'
    identity=dict(annotation=annotation_identity(args.annotation),motion_file=annotation_identity(args.motion_file),
                  policy=POLICY,rows=n,generator_sha256=sha256_file(__file__))
    if camera_manifest.exists():
        record=json.loads(camera_manifest.read_text())
        if record['contract']!=identity or not record['complete']:
            raise ValueError('Partial/different camera build; inspect it before choosing a new destination')
        for name in ('camera_R','camera_T'):
            if sha256_file(folder/(name+'.npy'))!=record['files'][name]:raise ValueError('Camera SHA mismatch')
        rmap=np.load(folder/'camera_R.npy',mmap_mode='r');tmap=np.load(folder/'camera_T.npy',mmap_mode='r')
    else:
        if any((folder/(x+'.npy')).exists() for x in ('camera_R','camera_T')):raise FileExistsError('Camera arrays already exist')
        save_json(camera_manifest,dict(contract=identity,complete=False))
        motions=torch.load(args.motion_file,map_location='cpu',weights_only=False)
        lookup={}
        for key,value in motions.items():
            scene=key.split('/')[-3];seq,subject=key.split('/')[-1].split('.mp4-',1)
            name=f'{scene}/{seq}/{subject}'
            if name in lookup:raise ValueError('Duplicate official subject identity')
            lookup[name]=value
        seqs=db['seqname'];bounds=[0]
        for first in range(0,n-1,16384):
            stop=min(first+16385,n);chunk=seqs[first:stop]
            bounds.extend((first+1+np.flatnonzero(chunk[1:]!=chunk[:-1])).tolist())
        bounds.append(n)
        missing=sorted(set(str(seqs[a]) for a in bounds[:-1])-set(lookup))
        if missing:raise ValueError('Missing official tracks: '+str(missing[:20]))
        rmap=np.lib.format.open_memmap(folder/'camera_R.npy',mode='w+',dtype='float64',shape=(n,3,3))
        tmap=np.lib.format.open_memmap(folder/'camera_T.npy',mode='w+',dtype='float64',shape=(n,3))
        changed=0;maximum=0.;same_camera_errors=[];camera_check={};mismatch_ext=0.
        for k,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
            name=str(seqs[a]);d=lookup[name];r,t=reconstruct_camera(d)
            ids=np.asarray([frame_number(x) for x in db['imgname'][a:b]],np.int64)
            if ids.min()<0 or ids.max()>=len(r) or np.any(np.diff(ids)<=0):raise ValueError('Frame identity mismatch: '+name)
            # The auxiliary BEDLAM transform must be row-identical, even though
            # its rotation is not itself a physical moving-camera rotation.
            error=float(np.abs(np.asarray(d['cam_ext'])[ids]-db['cam_ext'][a:b]).max())
            mismatch_ext=max(mismatch_ext,error)
            if error>2e-5:raise ValueError(f'Official/TRAM identity mismatch {name}: {error}')
            rmap[a:b]=r[ids];tmap[a:b]=t[ids]
            angles=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude()
            changed+=int((angles>1e-4).sum());maximum=max(maximum,float(angles.max()) if len(angles) else 0.)
            # Official world motion has a subject-specific constant SE(3)
            # placement gauge. Compare relative camera R/T, NOT absolute poses.
            # Rrel=R[t+1]R[t]^T; trel=t[t+1]-Rrel*t[t] are gauge invariant.
            video=name.rsplit('/',1)[0];common=min(5,len(r))
            relative_r=r[1:common]@r[:common-1].transpose(0,2,1)
            relative_t=t[1:common]-np.einsum('nij,nj->ni',relative_r,t[:common-1])
            if video in camera_check:
                r0,t0=camera_check[video];count=min(len(relative_r),len(r0))
                er=float(np.max(np.abs(relative_r[:count]-r0[:count])));et=float(np.max(np.abs(relative_t[:count]-t0[:count])))
                same_camera_errors.append([er,et])
            else:camera_check[video]=(relative_r.copy(),relative_t.copy())
            if k%2000==0:print('CAMERA',k,len(bounds)-1,'rows',b,'seconds',round(time.monotonic()-begin,1),flush=True)
        rmap.flush();tmap.flush()
        same=np.asarray(same_camera_errors)
        record=dict(contract=identity,complete=True,tracks=len(bounds)-1,nonzero_rotation_intervals=changed,
            max_rotation_step_deg=float(np.rad2deg(maximum)),auxiliary_cam_ext_max_abs=mismatch_ext,
            shared_camera_relative_max_abs=same.max(0).tolist(),shared_camera_relative_p99_abs=np.percentile(same,99,axis=0).tolist(),
            files={key:sha256_file(folder/(key+'.npy')) for key in ('camera_R','camera_T')})
        if changed==0:raise ValueError('No camera rotation variation reconstructed')
        # Degenerate axis-angle representations occasionally amplify roundoff;
        # investigate gross camera disagreement instead of training past it.
        if np.percentile(same[:,0],99)>.002 or np.percentile(same[:,1],99)>.03:
            raise ValueError('Cross-subject camera audit failed: '+str(record))
        save_json(camera_manifest,record);print('CAMERA_COMPLETE',json.dumps(record),flush=True)
        del motions,lookup,camera_check
    manifest=folder/'manifest.json'
    if manifest.exists():
        verify_label_manifest(folder,args.annotation);print('LABELS already complete',flush=True);return
    if any((folder/(x+'.npy')).exists() for x in ('label','mask','speed_mps')):raise FileExistsError('Partial labels already exist')
    geom=SMPLGeometry().to(args.device).eval()
    arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=d,shape=(n,6))
            for k,d in [('label','uint8'),('mask','bool'),('speed_mps','float32')]}
    old={k:np.load(Path(args.old_labels)/'bedlam_vid'/(k+'.npy'),mmap_mode='r') for k in ('label','mask','speed_mps')}
    changed=0;total=0;valid_count=np.zeros(6,np.int64);positive=np.zeros(6,np.int64)
    changed_moving=0;moving_total=0
    with torch.no_grad():
        for a in range(0,n,8192):
            b=min(a+8192,n);stop=min(b+1,n);rows=slice(a,stop)
            pose=torch.tensor(db['pose'][rows],dtype=torch.float32,device=args.device)
            beta=torch.tensor(db['shape'][rows],dtype=torch.float32,device=args.device)
            joints=geom.skeleton(pose,beta)[0][:,JOINT_IDS].cpu().numpy().astype(np.float64)
            xcam=joints+db['trans'][rows,None,:]+db['cam_ext'][rows,None,:3,3]
            world=np.einsum('nji,nkj->nki',rmap[rows],xcam-tmap[rows,None,:])
            ids=np.asarray([frame_number(x) for x in db['imgname'][rows]])
            valid=np.broadcast_to(~db['invalid'][rows,None],(stop-a,6)).copy()
            y,m,speed=labels_from_positions(world,valid,ids,db['seqname'][rows],30.,1)
            np.testing.assert_array_equal(old['mask'][a:b],m[:b-a])
            valid_count+=m[:b-a].sum(0);positive+=(m[:b-a] & (y[:b-a]>0)).sum(0)
            changed+=int((m[:b-a] & (old['label'][a:b]!=y[:b-a])).sum());total+=int(m[:b-a].sum())
            rotation=np.zeros(stop-a,bool)
            rotation[:-1]=np.max(np.abs(np.diff(rmap[rows],axis=0)),axis=(1,2))>1e-4
            mm=m[:b-a]&rotation[:b-a,None]
            changed_moving+=int((mm & (old['label'][a:b]!=y[:b-a])).sum());moving_total+=int(mm.sum())
            for k,v in [('label',y),('mask',m),('speed_mps',speed)]:arrays[k][a:b]=v[:b-a]
            if a//8192%64==0:print('LABEL',b,n,'changed',changed,'seconds',round(time.monotonic()-begin,1),flush=True)
    for arr in arrays.values():arr.flush()
    prev=json.loads((Path(args.old_labels)/'bedlam_vid/manifest.json').read_text())
    contract=dict(version=STATIC_VERSION,annotation=annotation_identity(args.annotation),rows=n,full_rows=n,
        joint_ids=JOINT_IDS,joint_names=JOINT_NAMES,threshold_mps=.15,original_fps=30.,frame_step=1,
        interval='t_to_t+1_last_masked',position_policy=POLICY,generator_sha256=sha256_file(__file__),
        geometry_sha256=sha256_file(ROOT/'lib/util/beyondretarget/smpl_geometry.py'),
        smpl_sha256=sha256_file(ROOT/'data/smpl/SMPL_NEUTRAL.pkl'),camera_manifest_sha256=sha256_file(camera_manifest))
    result=dict(contract=contract,**contract,complete=True,processed_rows=n,audit_only=False,
        files={k:sha256_file(folder/(k+'.npy')) for k in arrays},valid_count=valid_count.tolist(),
        positive_count=positive.tolist(),positive_rate=(positive/np.maximum(valid_count,1)).tolist(),
        projection_audit=prev['projection_audit'],label_changes_vs_v1=dict(changed=changed,valid=total,
            changed_fraction=changed/max(total,1),changed_moving=changed_moving,moving_valid=moving_total))
    save_json(manifest,result)
    # H36M's native measured fixed-camera labels remain byte-identical.
    target=root/'h36m_vid'
    if not target.exists():target.symlink_to(Path(args.old_labels).resolve()/'h36m_vid',target_is_directory=True)
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
