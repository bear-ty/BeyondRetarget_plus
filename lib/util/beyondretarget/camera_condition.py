"""Row-exact relative camera rotation, shared by static training and inference.

Input is WORLD-TO-CAMERA R. Output at t is R[t+1] @ R[t].T,
in pytorch3d row-6D convention. Repeat the last interval inside each native
16-frame window, exactly the same policy in training and video inference.
No GT body/root/velocity or camera translation enters the predictor.
"""
import numpy as np

CAMERA_CONDITION_VERSION = 'w2c_relative_rotation_rows6d_same16_v1'


def camera_relative_6d(world_to_camera):
    r = np.asarray(world_to_camera, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or len(r) < 2:
        raise ValueError('Camera conditioning requires a continuous [T,3,3] rotation window')
    if (not np.isfinite(r).all() or not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-3)):
        raise ValueError('Camera rotations must be finite and proper')
    delta = r[1:] @ r[:-1].transpose(0, 2, 1)
    delta = np.concatenate((delta, delta[-1:]), axis=0)
    return delta[:, :2, :].reshape(len(r), 6).astype(np.float32)


def continuous_window_frames(frame, valid=None, length=16):
    """Same gap policy and stride-one windows as official HMR_VIMO.inference."""
    frame = np.asarray(frame, dtype=np.int64)
    if valid is not None:
        frame = frame[np.asarray(valid, dtype=bool)]
    if len(frame) == 0:
        return []
    if np.any(np.diff(frame) <= 0):
        raise ValueError('Expected strictly increasing frame IDs')
    cuts = np.r_[0, np.flatnonzero(np.diff(frame) != 1) + 1, len(frame)]
    return [frame[start:start+length] for a, b in zip(cuts[:-1], cuts[1:])
            for start in range(int(a), int(b)-length+1)]


def stitch_native_windows(windows, frame_ids, width):
    """Official first-9 / center-1 / last-8; no temporal re-windowing."""
    import torch
    ids = np.asarray(frame_ids)
    cuts = np.r_[0, np.flatnonzero(np.diff(ids) != 1)+1, len(ids)]
    parts, cursor = [], 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = int(b-a)
        if n < 16:
            raise ValueError('No native window for a chunk shorter than 16')
        for start in range(n-15):
            if cursor >= len(windows) or windows[cursor].shape != (16, width):
                raise ValueError('Native window count/shape mismatch')
            offsets = (slice(None) if n == 16 else slice(0, 9) if start == 0
                       else slice(8, 16) if start == n-16 else slice(8, 9))
            parts.append(windows[cursor][offsets]); cursor += 1
    if cursor != len(windows):
        raise ValueError('Extra native windows')
    result = torch.cat(parts)
    if len(result) != len(ids):
        raise AssertionError('Frame/static association mismatch')
    return result
