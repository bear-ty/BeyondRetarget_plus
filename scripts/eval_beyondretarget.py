"""Evaluate camera-frame robot output against FRAME-MATCHED native SMPL labels.

GT NPZ: pose [N,72], shape/betas [N,10], optional frame [N], valid [N], fps.
This is the v5 semantic robot-pose metric, not official TRAM human evaluation.
World RTE requires separately verified world GT and is deliberately not faked.
Use the explicit .raw.npz sidecar (or --raw-only inference) here; default final
world-Zup postprocessed output is NOT a camera-frame prediction.
"""
import argparse,json,os,sys
from pathlib import Path
if not os.environ.get('PYTHONNOUSERSITE'):
    os.environ['PYTHONNOUSERSITE']='1';os.execv(sys.executable,[sys.executable,'-s',*sys.argv])
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from lib.core.config import update_cfg
from lib.util.beyondretarget.loss import TramGvrmrV5Loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg',default='configs/config_beyondretarget_g1.yaml')
    p.add_argument('--pred',required=True);p.add_argument('--gt',required=True)
    p.add_argument('--output');p.add_argument('--fps',type=float);p.add_argument('--frame-step',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(2)
    cfg=update_cfg(args.cfg);cfg.ROBOT.STAGE='head_only'
    prediction=dict(np.load(args.pred,allow_pickle=False));gt=dict(np.load(args.gt,allow_pickle=False))
    if str(prediction.get('coordinate_frame',''))!='camera':
        raise ValueError('Expected camera-frame raw output: use the .raw.npz sidecar or --raw-only; '
                         'do not evaluate default world-Zup postprocessed output as camera-frame motion')
    if str(prediction.get('root_6d_convention',''))!='pytorch3d_rows_smpl_root':
        raise ValueError('Unspecified root convention; refuse to guess')
    frame=np.asarray(prediction['frame'],dtype=np.int64);n=len(frame)
    if n==0 or np.any(np.diff(frame)<=0):raise ValueError('Prediction frame IDs must increase strictly')
    gt_frame=np.asarray(gt.get('frame',np.arange(len(gt['pose']))),dtype=np.int64)
    if len(np.unique(gt_frame))!=len(gt_frame):raise ValueError('GT contains duplicate frame/person keys')
    mapping={int(k):i for i,k in enumerate(gt_frame)}
    if any(int(k) not in mapping for k in frame):raise ValueError('Prediction frames are missing in GT; no silent truncation')
    ids=np.asarray([mapping[int(k)] for k in frame])
    fps=float(args.fps if args.fps is not None else np.asarray(prediction['fps']).item())
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid physical FPS')
    if 'fps' in gt and not np.allclose(gt['fps'],fps):raise ValueError('GT/prediction FPS mismatch')
    pose=np.asarray(gt['pose'])[ids]
    betas=np.asarray(gt['shape'] if 'shape' in gt else gt['betas'])[ids]
    valid=np.asarray(gt.get('valid',np.ones(len(gt_frame),bool)),dtype=bool)[ids]
    batch={'pose':torch.as_tensor(pose,dtype=torch.float32)[None],
           'betas':torch.as_tensor(betas,dtype=torch.float32)[None],
           'frame_mask':torch.from_numpy(valid)[None],'fps':torch.tensor([fps]),
           'continuous_pair_mask':torch.from_numpy(np.diff(frame)==args.frame_step)[None]}
    pred={k:torch.as_tensor(prediction[k],dtype=torch.float32)[None] for k in ('dof','root_pos','root_rot_6d')}
    loss=TramGvrmrV5Loss(cfg).eval()
    if str(prediction.get('robot_name'))!=loss.robot_loss.robot_spec.name:raise ValueError('Robot spec mismatch')
    with torch.no_grad():_,stats=loss(pred,batch,metrics=True)
    result={'frames':n,'valid_frames':int(valid.sum()),'fps':fps,
            'reference':'neutral posed SMPL mesh -> SMPL-24 -> fixed robot morphology and torso proxy',
            'metrics':{k:float(v) for k,v in stats.items()},
            'note':'Camera-frame semantic pose metrics; no world RTE without verified world GT.'}
    text=json.dumps(result,indent=2);print(text)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+'\n')


if __name__=='__main__':main()
