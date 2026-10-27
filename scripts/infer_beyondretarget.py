"""Video + one bbox track -> canonical C-RC1 world-Z-up robot motion."""
import argparse,json,os,sys,tempfile
from pathlib import Path
if not os.environ.get('PYTHONNOUSERSITE'):
    os.environ['PYTHONNOUSERSITE']='1';os.execv(sys.executable,[sys.executable,'-s',*sys.argv])
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import cv2
import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_rotation_6d,matrix_to_quaternion
from lib.core.config import update_cfg
from lib.models.tram_robot import TramRobotAdapter,load_human_weights,load_tram_robot_checkpoint
from lib.models.tram_motion_robot import build_robot_model
from lib.robot.robot_spec import load_robot_spec,model_dof_to_xml_dof
from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.root_basis import ROBOT_IN_SMPL,ROOT_BASIS_VERSION
from lib.util.beyondretarget.inference_policy import (
    plan_outputs,validate_request,add_postprocess_arguments,validate_postprocess_request,
)


def video_to_frames(video,out_dir):
    cap=cv2.VideoCapture(str(video));paths=[];fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps<=0:raise ValueError('Cannot read video/FPS')
    while True:
        ok,frame=cap.read()
        if not ok:break
        path=Path(out_dir)/f'{len(paths):08d}.jpg'
        if not cv2.imwrite(str(path),frame):raise IOError('Frame extraction failed: '+str(path))
        paths.append(str(path))
    cap.release()
    if not paths:raise ValueError('Empty video')
    return np.asarray(paths),fps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg',default='configs/config_beyondretarget_g1.yaml')
    p.add_argument('--video',required=True);p.add_argument('--boxes',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,help='FINAL postprocessed NPZ by default; raw only with --raw-only')
    p.add_argument('--device',default='cuda');p.add_argument('--camera-trajectory')
    p.add_argument('--static-checkpoint',help='Optional standalone shared static head; otherwise use the merged Plus checkpoint')
    p.add_argument('--raw-only',action='store_true',help='Explicitly skip height/static postprocessing; --output becomes raw')
    p.add_argument('--raw-output',help='Optional raw sidecar path; default <output-stem>.raw.npz')
    p.add_argument('--postprocess-device',default='cpu');p.add_argument('--clearance',type=float,default=None,
        help='Human-sole relative offset, default 0m; -0.04 adds optional 4cm downward shift (may penetrate)')
    p.add_argument('--smooth-alpha',type=float,default=0.,help='Optional DoF-only EMA; never modify frozen root')
    add_postprocess_arguments(p,'--postprocess-mode')
    args=p.parse_args();cfg=update_cfg(args.cfg)
    plan=plan_outputs(args.output,raw_only=args.raw_only,raw_output=args.raw_output)
    validate_request(plan,args.checkpoint,args.camera_trajectory,args.smooth_alpha)
    if plan.final is not None:
        validate_postprocess_request(args.postprocess_mode,video=args.video,boxes=args.boxes,
            keypoints=args.keypoints,confirm_single_shot=args.confirm_single_shot)
    for name in ('video','boxes','checkpoint','static_checkpoint','camera_trajectory'):
        if getattr(args,name) and not Path(getattr(args,name)).is_file():
            raise FileNotFoundError(f'--{name.replace("_","-")}: {getattr(args,name)}')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(int(os.environ.get('TRAM_CPU_THREADS','4')))
    spec=load_robot_spec(cfg.ROBOT.SPEC)
    model=build_robot_model(cfg,spec).to(device)
    checkpoint,_,_=load_tram_robot_checkpoint(model,args.checkpoint)
    embedded_human = checkpoint.get('human')
    human_path = Path(cfg.MODEL.CHECKPOINT)
    if embedded_human is not None:
        load_human_weights(model,payload={'human': embedded_human})
        human_sha = checkpoint.get('human_provenance',{}).get('sha256')
        if not human_sha:
            human_sha = checkpoint['provenance'].get('human_checkpoint_sha256')
        if not human_sha:
            raise ValueError('Unified checkpoint is missing human backbone provenance')
    else:
        if not human_path.is_file():
            raise FileNotFoundError(
                f'No embedded human backbone and configured TRAM/VIMO checkpoint is missing: {human_path}'
            )
        load_human_weights(model,human_path)
        human_sha = sha256_file(human_path)
    static_model=None;static_sha=None
    if args.static_checkpoint:
        from lib.models.tram_static import load_static_head
        static_sha=sha256_file(args.static_checkpoint)
        static_model,_=load_static_head(args.static_checkpoint,device,human_sha)
        if sha256_file(args.static_checkpoint)!=static_sha:
            raise RuntimeError('Static checkpoint changed during load; retry with a stable checkpoint copy')
    elif plan.final is not None:
        from lib.models.tram_static import load_static_head_from_payload
        static_model,_=load_static_head_from_payload(checkpoint,device,human_sha)
        static_sha=sha256_file(args.checkpoint)
    expected_human_sha = checkpoint['provenance'].get('human_checkpoint_sha256')
    if expected_human_sha and expected_human_sha != human_sha:
        raise ValueError('Checkpoint provenance mismatch: human_checkpoint_sha256')
    if checkpoint['provenance'].get('robot_asset_sha256') != sha256_file(spec.xml_path):
        raise ValueError('Checkpoint provenance mismatch: robot_asset_sha256')
    # Robot YAML comments and relocatable asset paths are intentionally allowed
    # to differ from the training snapshot; the referenced XML/MJCF is the
    # executable compatibility boundary and is checked above.
    spec_sha=checkpoint['provenance'].get('robot_spec_sha256')
    if spec_sha and spec_sha != sha256_file(cfg.ROBOT.SPEC):
        print('Warning: robot YAML hash differs from the training snapshot; XML asset hash matches.', file=sys.stderr)
    model.eval()
    with np.load(args.boxes,allow_pickle=False) as data:
        boxes=np.asarray(data['boxes'],dtype=np.float32)
        valid=np.asarray(data['valid'],dtype=bool) if 'valid' in data else np.ones(len(boxes),bool)
        frame=np.asarray(data['frame'],dtype=np.int64) if 'frame' in data else np.arange(len(boxes))
        focal=data['img_focal'].copy() if 'img_focal' in data else None
        center=data['img_center'].copy() if 'img_center' in data else None
    if boxes.shape!=(len(frame),4) or not np.array_equal(frame,np.arange(len(frame))):
        raise ValueError('bbox NPZ must have one row per original video frame, frame=0..N-1; use valid for missing targets')
    if not np.isfinite(boxes[valid]).all() or np.any(boxes[valid,2:]<=boxes[valid,:2]):
        raise ValueError('Invalid xyxy boxes')
    if plan.final is not None:
        if not valid.all() or focal is None or center is None:
            raise ValueError('C-RC1 requires a complete continuous target and explicit matching intrinsics; split gaps first')
    camera=None
    if args.camera_trajectory:
        camera=np.load(args.camera_trajectory,allow_pickle=True)
        camera=camera.item() if isinstance(camera,np.ndarray) else dict(camera)
        r,t=np.asarray(camera['world_cam_R']),np.asarray(camera['world_cam_T'])
        if r.shape!=(len(boxes),3,3) or t.shape!=(len(boxes),3):raise ValueError('Camera trajectory must match ALL video frames')
        if not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('Non-finite camera trajectory')
        if not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-3) or not np.allclose(np.linalg.det(r),1.,atol=1e-3):
            raise ValueError('world_cam_R is not a proper rotation')
    static_kwargs={}
    if static_model is not None and getattr(static_model,'requires_camera',False):
        if camera is None:raise ValueError('Camera-conditioned static inference requires --camera-trajectory, also in raw-only mode')
        static_kwargs['camera_world_to_cam_R']=np.asarray(camera['world_cam_R']).transpose(0,2,1)
    with tempfile.TemporaryDirectory(prefix='tram_robot_video_') as folder:
        files,fps=video_to_frames(args.video,folder)
        if len(files)!=len(boxes):raise ValueError('Video/bbox length mismatch')
        result=model.infer_sequence(files,boxes,focal,center,valid,frame,str(device),static_predictor=static_model,**static_kwargs)
    if result is None:raise ValueError('No continuous valid track of at least 16 frames')
    output={k:v.numpy() if torch.is_tensor(v) else v for k,v in result.items()}
    raw_dof=output['dof'].copy()
    if args.smooth_alpha:
        for i in range(1,len(raw_dof)):
            if output['frame'][i]-output['frame'][i-1]==1:
                output['dof'][i]=args.smooth_alpha*output['dof'][i]+(1-args.smooth_alpha)*output['dof'][i-1]
        output['dof_raw']=raw_dof
    trained_contact=checkpoint['stage']=='contact_only'
    if trained_contact:
        for key in ('static','contact'):
            output[key+'_prob']=torch.sigmoid(torch.from_numpy(output[key+'_logits'])).numpy()
    else:
        for key in ('static_logits','contact_logits'):output.pop(key,None)
    output['dof_xml']=model_dof_to_xml_dof(torch.from_numpy(output['dof']),spec).numpy()
    rcam=rotation_6d_to_matrix(torch.from_numpy(output['root_rot_6d'])).numpy()
    basis=np.asarray(ROBOT_IN_SMPL,dtype=np.float32)
    output['robot_root_rotmat_camera']=rcam@basis
    if args.camera_trajectory:
        r,t=np.asarray(camera['world_cam_R']),np.asarray(camera['world_cam_T'])
        if r.shape!=(len(boxes),3,3) or t.shape!=(len(boxes),3):raise ValueError('Camera trajectory must match ALL video frames')
        if not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('Non-finite camera trajectory')
        if not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-3) or not np.allclose(np.linalg.det(r),1.,atol=1e-3):
            raise ValueError('world_cam_R is not a proper rotation')
        ids=output['frame'];r,t=r[ids],t[ids]
        position=np.einsum('nij,nj->ni',r,output['root_pos'])+t
        orientation=r@rcam
        conversion=np.asarray([[1.,0,0],[0,0,1.],[0,-1.,0]],dtype=np.float32)
        output['root_pos_world_zup']=position@conversion
        root_native=conversion.T@orientation@basis
        output['root_rot_world_zup_6d']=matrix_to_rotation_6d(torch.from_numpy(root_native)).numpy()
        output['root_quat_world_zup_wxyz']=matrix_to_quaternion(torch.from_numpy(root_native)).numpy()
        output['camera_to_world_zup_R']=conversion.T@r
        output['camera_to_world_zup_T']=t@conversion
        if 'camera_tracking_valid' in camera:
            tracking=np.asarray(camera['camera_tracking_valid'],dtype=bool)
            if tracking.shape!=(len(boxes),):raise ValueError('Camera tracking validity/frame mismatch')
            output['camera_tracking_valid']=tracking[ids]
    output.update(fps=np.asarray(fps),robot_name=np.asarray(spec.name),coordinate_frame=np.asarray('camera'),
                  root_basis=np.asarray(ROOT_BASIS_VERSION),root_6d_convention=np.asarray('pytorch3d_rows_smpl_root'),
                  joint_names=np.asarray(spec.model_dof_order),contact_head_trained=np.asarray(trained_contact))
    output['human_checkpoint_sha256']=np.asarray(human_sha)
    output['architecture']=np.asarray(cfg.ROBOT.get('ARCHITECTURE','decoder46_v5'))
    output['model_context_version']=np.asarray(checkpoint['context_version'])
    output['robot_checkpoint_sha256']=np.asarray(sha256_file(args.checkpoint))
    output['postprocess_applied']=np.asarray(False)
    if focal is not None:output['img_focal']=np.broadcast_to(focal,(len(boxes),))[output['frame']].copy()
    if center is not None:output['img_center']=np.broadcast_to(center,(len(boxes),2))[output['frame']].copy()
    output['static_head_trained']=np.asarray(static_model is not None)
    if static_model is not None:
        output['static_checkpoint_sha256']=np.asarray(static_sha)
        output['static_version']=np.asarray(static_model.static_version)
    # Retain a clearly labelled raw sidecar. The primary --output is only
    # published by save_postprocessed AFTER the complete canonical pipeline.
    plan.raw.parent.mkdir(parents=True,exist_ok=True)
    with plan.raw.open('xb') as handle:np.savez_compressed(handle,**output)
    if plan.final is not None:
        from scripts.postprocess_beyondretarget import process_prediction,save_postprocessed
        final,report,stages=process_prediction(output,spec,args.postprocess_device,args.clearance,
            mode=args.postprocess_mode,video=args.video,boxes=args.boxes,keypoints=args.keypoints,
            confirm_single_shot=args.confirm_single_shot)
        report.setdefault('sources',{}).update(raw=str(plan.raw.resolve()),robot_spec=str(Path(cfg.ROBOT.SPEC).resolve()))
        report.setdefault('sha256',{}).update(raw=sha256_file(plan.raw),robot_spec=sha256_file(cfg.ROBOT.SPEC))
        save_postprocessed(plan.final,final,report,stages)
    print(json.dumps({'output':str(plan.primary),'raw_output':str(plan.raw),
                      'postprocess_applied':plan.final is not None,'static_head_trained':static_model is not None,
                      'postprocess_mode':args.postprocess_mode if plan.final is not None else None,
                      'frames':len(output['dof']),'fps':fps,
                      'world_motion_available':bool(args.camera_trajectory),'contact_head_trained':trained_contact,
                      'output_coordinate_frame':'world_zup' if plan.final is not None else 'camera',
                      'notice':'Raw exports are explicitly marked; only final exports have postprocess_applied=true.'},indent=2))


if __name__=='__main__':main()
