"""Row-exact labels; world speed (m/s), never root-relative displacement."""
import hashlib
import json
from pathlib import Path
import numpy as np
from lib.models.tram_static import STATIC_VERSION, JOINT_IDS


def annotation_identity(path):
    path = Path(path).resolve()
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


def labels_from_positions(position, valid, frames, tracks, original_fps, expected_step, threshold=.15):
    """Label t describes interval t -> t+1. Last row / gaps are masked."""
    n = len(position)
    if position.shape != (n, 6, 3) or valid.shape != (n, 6):
        raise ValueError('Expected [N,6,3] positions and [N,6] validity')
    speed = np.zeros((n, 6), np.float32)
    mask = np.zeros((n, 6), bool)
    dt = np.diff(frames) / float(original_fps)
    pair = (np.diff(frames) == expected_step) & (tracks[1:] == tracks[:-1])
    finite = np.isfinite(position).all(-1)
    mask[:-1] = (valid[:-1] & valid[1:] & finite[:-1] & finite[1:] & pair[:, None])
    velocity = np.linalg.norm(np.diff(position, axis=0), axis=-1) / np.maximum(dt[:, None], 1e-10)
    speed[:-1] = np.where(mask[:-1], velocity, 0)
    return (speed < threshold).astype(np.uint8), mask, speed


def verify_label_manifest(folder, annotation):
    path = Path(folder) / 'manifest.json'
    m = json.loads(path.read_text())
    if (not m.get('complete') or m.get('version') != STATIC_VERSION
            or tuple(m.get('joint_ids', ())) != JOINT_IDS
            or m['annotation'] != annotation_identity(annotation)):
        raise ValueError('Incomplete or mismatched stationary labels: ' + str(folder))
    from lib.util.beyondretarget.experiment import sha256_file
    for name,dtype in [('label','uint8'),('mask','bool'),('speed_mps','float32')]:
        file=Path(folder)/(name+'.npy')
        array=np.load(file,mmap_mode='r')
        if array.shape!=(m['rows'],6) or array.dtype!=np.dtype(dtype) or sha256_file(file)!=m['files'][name]:
            raise ValueError('Static label array shape/dtype/SHA mismatch: '+str(file))
    return m
