import contextlib
import io
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EMDB_SCRIPTS = ROOT / 'scripts' / 'emdb'


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _forbidden_initialization(*args, **kwargs):
    raise AssertionError('Model initialization must not be reached')


class _FakeSMPL:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, body_pose, **kwargs):
        n_frames = len(body_pose)
        return types.SimpleNamespace(
            vertices=torch.zeros(n_frames, 8, 3),
            joints=torch.zeros(n_frames, 24, 3),
        )


def _axis_angle_to_matrix(value):
    return torch.eye(3, dtype=value.dtype).repeat(len(value), 1, 1)


def _matrix_to_axis_angle(value):
    return torch.zeros(len(value), 3, dtype=value.dtype)


def _matrix_to_quaternion(value):
    result = torch.zeros(len(value), 4, dtype=value.dtype)
    result[:, 0] = 1
    return result


def _camera_eval_result(cam_t, cam_q, mean):
    trajectory = types.SimpleNamespace(
        positions_xyz=np.asarray(cam_t),
        orientations_quat_wxyz=np.asarray(cam_q),
    )
    return {'mean': mean}, trajectory, trajectory


def _eval_slam(pred_traj, cam_t, cam_q, correct_scale=False, **kwargs):
    return _camera_eval_result(cam_t, cam_q, 1.0 if correct_scale else 2.0)


def _make_stubs(forbid_smpl=True):
    smpl = _forbidden_initialization if forbid_smpl else _FakeSMPL
    eval_utils = _module(
        'lib.utils.eval_utils',
        SMPL=smpl,
        batch_align_by_pelvis=lambda values, pelvis_idxs: values,
        batch_compute_similarity_transform_torch=lambda pred, gt: pred,
        compute_error_accel=lambda joints_pred, joints_gt: np.zeros(len(joints_pred)),
        first_align_joints=lambda target, pred: pred,
        global_align_joints=lambda target, pred: pred,
        compute_jpe=lambda target, pred: np.zeros(len(target)),
        compute_rte=lambda target, pred: np.zeros(len(target)),
        computer_erve=lambda gt_ori, gt_j3d, pred_ori, pred_j3d: np.zeros(len(gt_j3d)),
    )
    eval_utils.__all__ = [
        'SMPL', 'batch_align_by_pelvis', 'batch_compute_similarity_transform_torch',
        'compute_error_accel', 'first_align_joints', 'global_align_joints',
        'compute_jpe', 'compute_rte', 'computer_erve',
    ]

    lazy_config = types.SimpleNamespace(load=_forbidden_initialization)
    stubs = {
        'cv2': _module('cv2'),
        'tqdm': _module('tqdm', tqdm=lambda values: values),
        'lib': _module('lib'),
        'lib.camera': _module(
            'lib.camera',
            run_metric_slam=_forbidden_initialization,
            align_cam_to_world=_forbidden_initialization,
        ),
        'lib.camera.slam_utils': _module('lib.camera.slam_utils', eval_slam=_eval_slam),
        'lib.pipeline': _module('lib.pipeline'),
        'lib.pipeline.tools': _module('lib.pipeline.tools', arrange_boxes=_forbidden_initialization),
        'lib.utils': _module('lib.utils'),
        'lib.utils.utils_detectron2': _module(
            'lib.utils.utils_detectron2', DefaultPredictor_Lazy=_forbidden_initialization,
        ),
        'lib.utils.eval_utils': eval_utils,
        'lib.utils.rotation_conversions': _module(
            'lib.utils.rotation_conversions',
            axis_angle_to_matrix=_axis_angle_to_matrix,
            matrix_to_axis_angle=_matrix_to_axis_angle,
            matrix_to_quaternion=_matrix_to_quaternion,
        ),
        'lib.vis': _module('lib.vis'),
        'lib.vis.traj': _module(
            'lib.vis.traj', traj_filter=lambda vertices, joints: (vertices, joints),
        ),
        'lib.models': _module('lib.models', get_hmr_vimo=_forbidden_initialization),
        'lib.datasets': _module('lib.datasets'),
        'lib.datasets.image_dataset': _module(
            'lib.datasets.image_dataset', ImageDataset=_forbidden_initialization,
        ),
        'segment_anything': _module(
            'segment_anything',
            SamPredictor=_forbidden_initialization,
            sam_model_registry={'vit_h': _forbidden_initialization},
        ),
        'detectron2': _module('detectron2'),
        'detectron2.config': _module('detectron2.config', LazyConfig=lazy_config),
    }
    for package in ['lib', 'lib.camera', 'lib.pipeline', 'lib.utils', 'lib.vis', 'lib.datasets', 'detectron2']:
        stubs[package].__path__ = []
    return stubs


def _run_entry(script, args, stubs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    argv = [str(EMDB_SCRIPTS / script), *map(str, args)]
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(sys, 'argv', argv), \
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            runpy.run_path(str(EMDB_SCRIPTS / script), run_name='__main__')
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        else:
            code = 0
    return code, stdout.getvalue(), stderr.getvalue()


def _entry_args(script, data_root, output_root):
    if script == 'run_eval.py':
        return ['--data-root', data_root, '--input_dir', output_root]
    return ['--data-root', data_root, '--output_dir', output_root]


def _write_sequence(data_root, selected=True, n_frames=5, valid=True):
    sequence = data_root / 'P0' / 'S0'
    sequence.mkdir(parents=True)
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    annotation = {
        'emdb2': selected,
        'good_frames_mask': np.full(n_frames, valid, dtype=bool),
        'gender': 'neutral',
        'n_frames': n_frames,
        'camera': {
            'extrinsics': extrinsics,
            'intrinsics': np.eye(3, dtype=np.float32),
        },
        'smpl': {
            'poses_body': np.zeros((n_frames, 69), dtype=np.float32),
            'poses_root': np.zeros((n_frames, 3), dtype=np.float32),
            'betas': np.zeros(10, dtype=np.float32),
            'trans': np.zeros((n_frames, 3), dtype=np.float32),
        },
    }
    with (sequence / 'P0_S0_data.pkl').open('wb') as handle:
        pickle.dump(annotation, handle)


def _write_predictions(input_dir, n_frames=5):
    camera_dir = input_dir / 'camera'
    smpl_dir = input_dir / 'smpl'
    camera_dir.mkdir(parents=True)
    smpl_dir.mkdir(parents=True)
    np.savez(
        camera_dir / 'S0.npz',
        pred_cam_R=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        pred_cam_T=np.zeros((n_frames, 3), dtype=np.float32),
    )
    np.savez(
        smpl_dir / 'S0.npz',
        pred_rotmat=np.tile(np.eye(3, dtype=np.float32), (n_frames, 24, 1, 1)),
        pred_shape=np.zeros((n_frames, 10), dtype=np.float32),
        pred_trans=np.zeros((n_frames, 3), dtype=np.float32),
    )


class EMDBFailFastTests(unittest.TestCase):
    def test_empty_data_root_rejected_before_output_or_model_initialization(self):
        for script in ['run_cam.py', 'run_smpl.py', 'run_eval.py']:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / 'missing-data'
                output_root = root / 'output'
                code, _, stderr = _run_entry(
                    script, _entry_args(script, data_root, output_root), _make_stubs(),
                )
                self.assertEqual(code, 2)
                self.assertIn('No EMDB sequences selected', stderr)
                self.assertFalse(output_root.exists())

    def test_empty_split_rejected_before_output_or_model_initialization(self):
        for script in ['run_cam.py', 'run_smpl.py', 'run_eval.py']:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / 'data'
                output_root = root / 'output'
                _write_sequence(data_root, selected=False)
                code, _, stderr = _run_entry(
                    script, _entry_args(script, data_root, output_root), _make_stubs(),
                )
                self.assertEqual(code, 2)
                self.assertIn('No EMDB sequences selected', stderr)
                self.assertFalse(output_root.exists())

    def test_missing_predictions_are_reported_before_smpl_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / 'data'
            input_dir = root / 'results'
            _write_sequence(data_root)
            code, _, stderr = _run_entry(
                'run_eval.py', _entry_args('run_eval.py', data_root, input_dir), _make_stubs(),
            )
            self.assertEqual(code, 2)
            self.assertIn('Missing required EMDB predictions', stderr)
            self.assertIn(str(input_dir / 'camera' / 'S0.npz'), stderr)
            self.assertIn(str(input_dir / 'smpl' / 'S0.npz'), stderr)
            self.assertFalse((input_dir / 'evaluation.xlsx').exists())

    def test_zero_valid_human_sequences_does_not_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / 'data'
            input_dir = root / 'results'
            _write_sequence(data_root, valid=False)
            _write_predictions(input_dir)
            stubs = _make_stubs(forbid_smpl=False)
            camera_calls = []
            original_eval_slam = stubs['lib.camera.slam_utils'].eval_slam

            def recording_eval_slam(*args, **kwargs):
                camera_calls.append((args, kwargs))
                return original_eval_slam(*args, **kwargs)

            stubs['lib.camera.slam_utils'].eval_slam = recording_eval_slam
            writes = []
            with mock.patch.object(
                pd.DataFrame, 'to_excel',
                new=lambda df, path, index=False: writes.append((df.copy(), path)),
            ):
                code, stdout, stderr = _run_entry(
                    'run_eval.py', _entry_args('run_eval.py', data_root, input_dir), stubs,
                )
            self.assertEqual(code, 2)
            self.assertIn('Skipping S0: no valid EMDB frames', stdout)
            self.assertIn('No valid human EMDB sequences', stderr)
            self.assertEqual(camera_calls, [])
            self.assertEqual(writes, [])

    def test_non_finite_metrics_do_not_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / 'data'
            input_dir = root / 'results'
            _write_sequence(data_root)
            _write_predictions(input_dir)
            stubs = _make_stubs(forbid_smpl=False)

            def nan_eval_slam(pred_traj, cam_t, cam_q, correct_scale=False, **kwargs):
                return _camera_eval_result(cam_t, cam_q, float('nan'))

            stubs['lib.camera.slam_utils'].eval_slam = nan_eval_slam
            writes = []
            with mock.patch.object(
                pd.DataFrame, 'to_excel',
                new=lambda df, path, index=False: writes.append((df.copy(), path)),
            ):
                code, _, stderr = _run_entry(
                    'run_eval.py', _entry_args('run_eval.py', data_root, input_dir), stubs,
                )
            self.assertEqual(code, 2)
            self.assertIn('Non-finite EMDB evaluation metrics: ate, ate_s', stderr)
            self.assertEqual(writes, [])

    def test_valid_evaluation_still_writes_finite_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / 'data'
            input_dir = root / 'results'
            _write_sequence(data_root)
            _write_predictions(input_dir)
            writes = []
            with mock.patch.object(
                pd.DataFrame, 'to_excel',
                new=lambda df, path, index=False: writes.append((df.copy(), path)),
            ):
                code, _, stderr = _run_entry(
                    'run_eval.py',
                    _entry_args('run_eval.py', data_root, input_dir),
                    _make_stubs(forbid_smpl=False),
                )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(writes), 1)
            dataframe, output_path = writes[0]
            self.assertEqual(output_path, str(input_dir / 'evaluation.xlsx'))
            self.assertEqual(set(dataframe['Metric']), {
                'pa_mpjpe', 'mpjpe', 'pve', 'accel', 'wa_mpjpe',
                'w_mpjpe', 'rte', 'erve', 'ate', 'ate_s',
            })
            self.assertTrue(np.isfinite(dataframe['Value'].to_numpy(dtype=float)).all())
            self.assertEqual(dataframe.set_index('Metric').loc['ate', 'Value'], 1.0)
            self.assertEqual(dataframe.set_index('Metric').loc['ate_s', 'Value'], 2.0)

    def test_run_sh_fail_fast_and_success_exit_codes(self):
        for status, expected_calls in [(7, 1), (0, 3)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bin_dir = root / 'bin'
                bin_dir.mkdir()
                log = root / 'calls.log'
                fake_python = bin_dir / 'python'
                fake_python.write_text(
                    '#!/bin/sh\n'
                    'printf "%s\\n" "$*" >> "$EMDB_TEST_LOG"\n'
                    'exit "$EMDB_TEST_PYTHON_STATUS"\n'
                )
                fake_python.chmod(0o755)
                env = os.environ.copy()
                env.update({
                    'PATH': str(bin_dir) + os.pathsep + env['PATH'],
                    'EMDB_TEST_LOG': str(log),
                    'EMDB_TEST_PYTHON_STATUS': str(status),
                })
                completed = subprocess.run(
                    ['bash', str(EMDB_SCRIPTS / 'run.sh')],
                    cwd=ROOT, env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, status, completed.stderr)
                calls = log.read_text().splitlines()
                self.assertEqual(len(calls), expected_calls)
                self.assertIn('run_cam.py', calls[0])
                if status == 0:
                    self.assertIn('run_smpl.py', calls[1])
                    self.assertIn('run_eval.py', calls[2])


if __name__ == '__main__':
    unittest.main()
