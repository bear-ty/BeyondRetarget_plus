"""Local background-compensated ankle motion. No action/clip-specific rules.

Homography is an approximate image-motion model, NOT a proof of world contact.
Work in a 0.2s local reference to avoid long-video homography-chain drift.
"""
import cv2
import numpy as np


def project(points,h):
    p=np.concatenate((points,np.ones((*points.shape[:-1],1))),axis=-1)@h.T
    if np.any(np.abs(p[...,-1])<1e-8):raise ValueError('Degenerate homography projection')
    return p[...,:2]/p[...,-1:]


def estimate_background(video,boxes):
    boxes=np.asarray(boxes)
    if boxes.ndim!=2 or boxes.shape[1:]!=(4,) or len(boxes)<1 or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
        raise ValueError('Expected nonempty finite xyxy boxes')
    cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS))
    ok,frame=cap.read()
    if not ok or not np.isfinite(fps) or fps<=0:
        cap.release();raise ValueError('Cannot decode video/FPS')
    previous=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    pairs=[];valid=[];errors=[];counts=[]
    for i in range(len(boxes)-1):
        ok,frame=cap.read()
        if not ok:
            cap.release();raise ValueError('Video/bbox length mismatch')
        current=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if current.shape!=previous.shape:
            cap.release();raise ValueError('Resolution changes; split shots first')
        h,w=previous.shape;mask=np.full((h,w),255,np.uint8)
        box=boxes[i];pad=.12*max(box[2]-box[0],box[3]-box[1])
        x1,y1,x2,y2=np.round(box+[-pad,-pad,pad,pad]).astype(int)
        mask[np.clip(y1,0,h):np.clip(y2,0,h),np.clip(x1,0,w):np.clip(x2,0,w)]=0
        pts=cv2.goodFeaturesToTrack(previous,600,.01,5,mask=mask)
        H=None;good=False;error=np.inf;count=0
        if pts is not None and len(pts)>=30:
            dest,st,_=cv2.calcOpticalFlowPyrLK(previous,current,pts,None,winSize=(21,21),maxLevel=3)
            if dest is not None:
                back,stb,_=cv2.calcOpticalFlowPyrLK(current,previous,dest,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    keep=(st[:,0]>0)&(stb[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=-1)<1.)
                    p,q=pts[:,0][keep],dest[:,0][keep]
                    if len(p)>=30:
                        H,inliers=cv2.findHomography(p,q,cv2.RANSAC,2.)
                        if H is not None and inliers is not None:
                            inside=inliers[:,0]>0;count=int(inside.sum())
                            if count:
                                error=float(np.median(np.linalg.norm(project(p[inside],H)-q[inside],axis=-1)))
                            good=(count>=30 and inside.mean()>.45 and error<1.5
                                  and np.isfinite(H).all() and np.linalg.cond(H)<1e5)
        if not good:H=np.eye(3)
        pairs.append(H);valid.append(good);errors.append(error);counts.append(count)
        previous=current
    extra,_=cap.read();cap.release()
    if extra:raise ValueError('Video contains extra frames; explicitly trim/remap inputs')
    return dict(pair_H=np.asarray(pairs,dtype=float).reshape(-1,3,3),pair_valid=np.asarray(valid,dtype=bool),
                pair_residual_px=np.asarray(errors),pair_inliers=np.asarray(counts),fps=np.asarray(fps))


def evidence_from_pairs(keypoints,boxes,pair_h,pair_valid,fps):
    kp=np.asarray(keypoints);boxes=np.asarray(boxes);n=len(kp)
    if kp.shape!=(n,17,3) or boxes.shape!=(n,4) or np.shape(pair_h)!=(n-1,3,3) or np.shape(pair_valid)!=(n-1,):
        raise ValueError('Expected aligned COCO17 observations, xyxy boxes, and consecutive homographies')
    if not np.isfinite(fps) or fps<=0 or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
        raise ValueError('Invalid image dimensions/FPS')
    speed=np.full((n,2),np.inf);spread=speed.copy();evidence=np.zeros((n,2),bool)
    radius=max(2,int(round(.1*fps)))
    for i in range(n-1):
        a,b=max(0,i-radius),min(n,i+radius+1)
        if b-a<5 or not np.asarray(pair_valid[a:b-1],bool).all():continue
        rows=np.arange(a,b);u,v=np.triu_indices(len(rows),k=2)
        local=[np.eye(3)]
        for h in pair_h[a:b-1]:
            mat=h@local[-1]
            if not np.isfinite(mat).all() or abs(mat[2,2])<1e-8:break
            local.append(mat/mat[2,2])
        if len(local)!=len(rows):continue
        try:
            pix=np.stack([project(kp[row,[15,16],:2],local[i-a]@np.linalg.inv(local[row-a])) for row in rows])
        except (ValueError,np.linalg.LinAlgError):continue
        slope=(pix[v]-pix[u])/((rows[v]-rows[u])/fps)[:,None,None]
        velocity=np.median(slope,axis=0)
        uncertainty=np.percentile(np.linalg.norm(slope-velocity[None],axis=-1),75,axis=0)
        scale=max(50.,float(np.median(boxes[rows,3]-boxes[rows,1])))
        speed[i]=np.linalg.norm(velocity,axis=-1)/scale;spread[i]=uncertainty/scale
        feet=kp[rows][:,[15,16]]
        evidence[i]=np.isfinite(feet).all(axis=(0,2)) & (feet[:,:,2].min(0)>.5)
    return dict(speed_bodyheight_s=speed,uncertainty_bodyheight_s=spread,evidence=evidence)
