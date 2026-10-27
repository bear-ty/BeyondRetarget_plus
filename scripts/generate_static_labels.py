"""CPU-only resumable stationary labels from native, row-matched annotations.

BEDLAM: exact native SMPL FK; Xcam=Jsmpl+trans+cam_ext.translation,
Xworld=Rcam.T@(Jsmpl+trans). Audited against provided image projections.
H36M: native mocap ankles/wrists in a FIXED camera; speed norm is world-
invariant. Native archive has no measured toe joints: MASK toe channels,
never manufacture toe stationarity from camera-relative SMPL fits.
3DPW: no verified world translation here; deliberately not a static source.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
from lib.datasets.robot_video_dataset import MappedNPZ, SOURCES, frame_number
from lib.models.tram_static import STATIC_VERSION, JOINT_IDS, JOINT_NAMES
from lib.util.beyondretarget.static_labels import annotation_identity, labels_from_positions
from lib.util.beyondretarget.smpl_geometry import SMPLGeometry
from lib.util.beyondretarget.experiment import sha256_file


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); os.replace(tmp, path)


def projection_audit(db, source, count=128):
    ids = np.linspace(0, db.meta['imgname'][0][0]-1, count, dtype=np.int64)
    if source == 'bedlam_vid':
        from lib.models.smpl import SMPL
        model = SMPL()
        p = torch.tensor(db['pose'][ids], dtype=torch.float32); b = torch.tensor(db['shape'][ids], dtype=torch.float32)
        with torch.no_grad():
            xyz = model(global_orient=p[:, :3], body_pose=p[:, 3:], betas=b).joints[:, 25:].numpy()
        xyz += db['trans'][ids, None, :] + db['cam_ext'][ids, None, :3, 3]
        valid = (db['part'][ids, :, 2] > 0) & ~db['invalid'][ids, None].astype(bool)
        rot = db['cam_ext'][ids, :3, :3]
        if not np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=1e-4):
            raise ValueError('BEDLAM camera rotations invalid')
    else:
        xyz = db['S'][ids, :, :3]
        valid = (db['S'][ids, :, 3] > 0) & (db['part'][ids, :, 2] > 0)
    valid &= xyz[..., 2] > .1
    uv = xyz[..., :2] / np.maximum(xyz[..., 2:], 1e-8)
    uv = uv * db['img_focal'][ids, None, None] + db['img_center'][ids, None, :]
    err = np.linalg.norm(uv - db['part'][ids, :, :2], axis=-1)[valid]
    result = dict(samples=count, median_px=float(np.median(err)), p95_px=float(np.percentile(err, 95)), max_px=float(err.max()))
    # H36M annotated pixels include lens distortion; native S remains absolute.
    limit = .1 if source == 'bedlam_vid' else 15.
    if result['p95_px'] > limit:
        raise ValueError(f'Coordinate / projection audit failed: {source}: {result}')
    return result


def generate(args, source, geom):
    db = MappedNPZ(Path(args.dataset_root) / 'dataset_ann' / SOURCES[source][0])
    total = db.meta['imgname'][0][0]
    n = min(total, args.limit) if args.limit else total
    folder = Path(args.output) / source
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    contract = dict(version=STATIC_VERSION, annotation=annotation_identity(db.path), rows=n, full_rows=total,
        joint_ids=JOINT_IDS, joint_names=JOINT_NAMES, threshold_mps=.15, original_fps=SOURCES[source][2],
        frame_step=SOURCES[source][3], interval='t_to_t+1_last_masked',
        generator_sha256=sha256_file(__file__), geometry_sha256=sha256_file(ROOT/'lib/util/beyondretarget/smpl_geometry.py'),
        smpl_sha256=sha256_file(ROOT/'data/smpl/SMPL_NEUTRAL.pkl'),
        position_policy='native_smpl_absolute_world' if source=='bedlam_vid' else 'native_mocap_fixed_camera_ankles_wrists_toes_masked')
    contract = json.loads(json.dumps(contract))
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if state and state['contract'] != contract:
        raise ValueError('Existing labels have a different contract; use a new output folder')
    if state.get('complete'):
        print(source, 'already complete', flush=True); return
    done = state.get('processed_rows', 0)
    mode = 'r+' if state else 'w+'
    arrays = {key:np.lib.format.open_memmap(folder/(key+'.npy'), mode=mode, dtype=dtype, shape=(n,6))
              for key,dtype in [('label','uint8'),('mask','bool'),('speed_mps','float32')]}
    audit = state.get('projection_audit') or projection_audit(db, source)
    print(source, 'projection audit', audit, flush=True)
    state.update(contract=contract, **{k:v for k,v in contract.items() if k not in ('contract',)},
                 complete=False, processed_rows=done, projection_audit=audit)
    atomic_json(manifest_path, state)
    begin = time.time()
    for first in range(done, n, args.chunk):
        end = min(n, first+args.chunk); stop = min(n, end+1)
        rows = slice(first, stop); count = stop-first
        images = db['imgname'][rows]
        frames = np.asarray([frame_number(x) for x in images], dtype=np.int64)
        tracks = db['seqname'][rows] if 'seqname' in db else np.asarray([str(x).rsplit('_',1)[0] for x in images])
        valid = np.ones((count, 6), bool)
        if source == 'bedlam_vid':
            pose = torch.tensor(db['pose'][rows], dtype=torch.float32)
            beta = torch.tensor(db['shape'][rows], dtype=torch.float32)
            with torch.no_grad():
                joints = geom.skeleton(pose, beta)[0][:, JOINT_IDS].numpy()
            cam = db['cam_ext'][rows]
            positions = np.einsum('nji,nkj->nki', cam[:, :3,:3], joints+db['trans'][rows,None,:])
            valid &= ~db['invalid'][rows,None].astype(bool)
            if not np.allclose(np.linalg.det(cam[:, :3,:3]), 1., atol=1e-3):
                raise ValueError(f'Invalid extrinsic rotations at {first}')
        else:
            s = db['S'][rows]
            positions = np.zeros((count,6,3), np.float32)
            # SPIN-24: L ankle5, R ankle0, L wrist11, R wrist6.
            positions[:,[0,2,4,5]] = s[:,[5,0,11,6],:3]
            valid[:] = False
            valid[:,[0,2,4,5]] = s[:,[5,0,11,6],3] > 0
        label, mask, speed = labels_from_positions(positions,valid,frames,tracks,SOURCES[source][2],SOURCES[source][3])
        for key,value in [('label',label),('mask',mask),('speed_mps',speed)]:
            arrays[key][first:end] = value[:end-first]; arrays[key].flush()
        state['processed_rows'] = end
        atomic_json(manifest_path, state)
        if (first-done)//args.chunk % 32 == 0 or end==n:
            print(f'{source}: {end:,}/{n:,} rows; {(end-done)/max(time.time()-begin,1):.0f} rows/s',flush=True)
    valid_count = np.zeros(6,np.int64); positive = valid_count.copy()
    identity = hashlib.sha256()
    for first in range(0,n,args.chunk):
        end = min(n,first+args.chunk); m=arrays['mask'][first:end]; y=arrays['label'][first:end]
        valid_count += m.sum(0); positive += (m & (y>0)).sum(0)
        identity.update(np.asarray(db['imgname'][first:end]).tobytes())
        if 'seqname' in db: identity.update(np.asarray(db['seqname'][first:end]).tobytes())
    state.update(complete=n==total, audit_only=n!=total, valid_count=valid_count.tolist(),
        positive_count=positive.tolist(), positive_rate=(positive/np.maximum(valid_count,1)).tolist(),
        row_identity_sha256=identity.hexdigest(), files={k:sha256_file(folder/(k+'.npy')) for k in arrays})
    atomic_json(manifest_path,state)
    print(json.dumps({k:state[k] for k in ('complete','valid_count','positive_rate')},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',default=os.environ.get('TRAM_DATA_ROOT','data/datasets'))
    p.add_argument('--output',default='outputs/static_labels/world_static6_v1')
    p.add_argument('--sources',default='bedlam_vid,h36m_vid')
    p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--limit',type=int,default=0,help='Audit only; cannot train with partial labels')
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.chunk<1 or args.limit<0:raise ValueError('Invalid chunk/limit')
    geom=SMPLGeometry().eval()
    for source in args.sources.split(','):
        if source not in ('bedlam_vid','h36m_vid'):raise ValueError('No verified world labels for '+source)
        generate(args,source,geom)


if __name__=='__main__':main()
