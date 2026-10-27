"""Pure filtering helpers copied from GVRMR; quaternion wxyz, 6D rows."""
import numpy as np
from typing import Dict
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

def _odd_window(length: int, requested: int) -> int:
    window = max(3, int(requested))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else length - 1
    return max(3, window)


def _filter_array(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    if values.shape[0] < 5:
        return values.astype(np.float32, copy=True)
    win = _odd_window(values.shape[0], window)
    order = min(int(poly), win - 1)
    filtered = savgol_filter(values, win, order, axis=0, mode="interp")
    return filtered.astype(np.float32)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    for idx in range(1, quat.shape[0]):
        if float(np.sum(quat[idx - 1] * quat[idx])) < 0.0:
            quat[idx] *= -1.0
    return quat.astype(np.float32)


def _filter_quat_wxyz(quat_wxyz: np.ndarray, window: int, poly: int) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    if quat.shape[0] < 5:
        return quat
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    filtered_xyzw = _filter_array(quat_xyzw, window, poly)
    filtered_wxyz = filtered_xyzw[:, [3, 0, 1, 2]]
    return _normalize_quat_wxyz(filtered_wxyz)


def _quat_wxyz_to_rot6d_mte(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    matrix = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    return matrix[:, :2, :].reshape(quat_wxyz.shape[0], 6).astype(np.float32)


def _filter_motion(data: Dict[str, np.ndarray], root_window: int, rot_window: int, dof_window: int, poly: int) -> Dict[str, np.ndarray]:
    quat = _filter_quat_wxyz(np.asarray(data["root_rot_quat"], dtype=np.float32), rot_window, poly)
    return {
        "fps": np.asarray(data["fps"], dtype=np.float64),
        "root_trans": _filter_array(np.asarray(data["root_trans"], dtype=np.float32), root_window, poly),
        "root_rot_quat": quat,
        "root_rot_6d": _quat_wxyz_to_rot6d_mte(quat),
        "dof": _filter_array(np.asarray(data["dof"], dtype=np.float32), dof_window, poly),
    }
