"""TRAM official SAM/masked DROID/ZoeDepth/SPEC using an existing target bbox track.

The detector/identity track is reused, not the old model's pose or trajectory.
All saved cameras have metric scale and the official gravity alignment.
"""
import argparse, gc, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'thirdparty/DROID-SLAM'))
sys.path.insert(0,str(ROOT/'thirdparty/DROID-SLAM/thirdparty/lietorch'))
# ZoeDepth/MiDaS requires legacy timm; isolate it to this camera subprocess.
# Robot inference/training keeps its original timm and checkpoint semantics.
if os.environ.get('TRAM_CAMERA_DEPS'):
    sys.path.insert(0,os.environ['TRAM_CAMERA_DEPS'])
import cv2
import numpy as np
import torch
from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.video_tracks import load_box_track

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--boxes',help='Single-person NPZ from scripts/extract_video_boxes.py')
    source.add_argument('--bbox-pt',help='Legacy PT dictionary containing bbx_xyxy')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--camera-mode',choices=['moving','static','auto'],default='moving')
    args=p.parse_args()
    bbox_source=args.boxes or args.bbox_pt
    if args.boxes:
        track=load_box_track(args.boxes,require_complete=True)
        boxes=track['boxes'];track_valid=track['valid']
        bbox_payload={'bbx_xyxy':torch.from_numpy(boxes)}
    else:
        bbox_payload=torch.load(args.bbox_pt,map_location='cpu',weights_only=False)
        boxes=bbox_payload['bbx_xyxy'].detach().cpu().numpy().astype(np.float32)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all():
            raise ValueError('bbx_xyxy must be a finite [T,4] tensor')
        track_valid=np.ones(len(boxes),dtype=bool)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    frames=out/'images';frames.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(args.video);fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,img=cap.read()
        if not ok:break
        dest=frames/f'{n:06d}.jpg'
        if not dest.exists() and not cv2.imwrite(str(dest),img):raise IOError(dest)
        n+=1
    cap.release();assert n==len(boxes) and n>=16,(n,len(boxes))
    assert len(list(frames.glob('*.jpg')))==n
    imgfiles=sorted(str(x) for x in frames.glob('*.jpg'))
    if not (out/'human_masks.npy').exists():
        from segment_anything import SamPredictor,sam_model_registry
        print('SAM foreground masks from verified bbox track',flush=True)
        model=sam_model_registry['vit_h'](checkpoint=str(ROOT/'data/pretrain/sam_vit_h_4b8939.pth')).cuda().eval()
        predictor=SamPredictor(model);masks=[]
        with torch.inference_mode():
            for i,file in enumerate(imgfiles):
                predictor.set_image(cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB))
                mask_boxes=bbox_payload.get('all_boxes',[boxes[j:j+1] for j in range(n)])[i]
                mask_boxes=np.concatenate([np.asarray(mask_boxes).reshape(-1,4),boxes[i:i+1]])
                mask_list=[]
                for bb in mask_boxes:
                    mask,_,_=predictor.predict(box=bb,multimask_output=False);mask_list.append(mask[0])
                masks.append(np.logical_or.reduce(mask_list))
                if i%30==0:print(f'SAM {i+1}/{n}',flush=True)
        np.save(out/'human_masks.npy',np.stack(masks))
        del predictor,model,masks;gc.collect();torch.cuda.empty_cache()
    # Official depth-scale estimation uses cv2.resize (does not accept bool).
    # Keep the saved binary masks, pass 0/1 float masks to the official frontend.
    masks=torch.from_numpy(np.load(out/'human_masks.npy')).float()
    if not (out/'camera.npy').exists():
        from lib.camera import calibrate_intrinsics,run_metric_slam,align_cam_to_world
        print('Official TRAM intrinsic search + metric masked SLAM',flush=True)
        intrinsics_cache=out/'intrinsics.npz'
        if intrinsics_cache.exists():
            with np.load(intrinsics_cache) as cache:
                assert str(cache['mode'])==args.camera_mode
                calib=cache['calib'];detected_static=bool(cache['detected_static'])
        else:
            calib,detected_static=calibrate_intrinsics(str(frames),masks,is_static=args.camera_mode=='static')
            np.savez(intrinsics_cache,calib=calib,detected_static=detected_static,mode=args.camera_mode)
        is_static=detected_static if args.camera_mode=='auto' else args.camera_mode=='static'
        r,t=run_metric_slam(str(frames),masks=masks,calib=calib,is_static=is_static)
        wr,wt,spec_f=align_cam_to_world(imgfiles[0],r,t)
        camera=dict(pred_cam_R=r.numpy(),pred_cam_T=t.numpy(),world_cam_R=wr.numpy(),world_cam_T=wt.numpy(),
                    img_focal=np.asarray(calib[0]),img_center=np.asarray(calib[2:]),spec_focal=spec_f)
        assert np.isfinite(camera['world_cam_T']).all() and np.isfinite(camera['world_cam_R']).all()
        np.save(out/'camera.npy',camera)
        (out/'camera_provenance.json').write_text(json.dumps(dict(
            requested_mode=args.camera_mode,auto_detected_static=bool(detected_static),used_static=bool(is_static),
            method='official_TRAM_masked_DROID_ZoeDepth_SPEC',segmentation='official_SAM_vit_h_target_bbox',
            bbox_source=str(Path(bbox_source).resolve()),bbox_sha256=sha256_file(bbox_source),
            input_video_sha256=sha256_file(args.video),frames=n,fps=fps,
            checkpoint_sha256={name:sha256_file(ROOT/'data/pretrain'/name) for name in
                               ['droid.pth','camcalib_sa_biased_l2.ckpt','sam_vit_h_4b8939.pth']}
        ),indent=2))
    camera=np.load(out/'camera.npy',allow_pickle=True).item()
    np.savez_compressed(out/'boxes.npz',boxes=boxes,valid=track_valid,frame=np.arange(n),
                        img_focal=camera['img_focal'],img_center=camera['img_center'])
    print(json.dumps(dict(complete=True,frames=n,fps=fps,output=str(out)),indent=2),flush=True)

if __name__=='__main__':main()
