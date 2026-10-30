"""CPU-only synthetic regressions for camera cache identity and validation."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'brplus_camera_cache_offline', ROOT / 'scripts/prepare_robot_video_camera.py',
)
camera_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera_script)


FRAME_COUNT = 16
FRAME_SHAPE = (6, 8)
CALIB = np.asarray([8.0, 8.0, 4.0, 3.0], dtype=np.float32)


def _write_track(path, offset=0.0):
    boxes = np.tile([0.0, 0.0, 4.0, 4.0], (FRAME_COUNT, 1)).astype(np.float32)
    boxes[:, 0] += offset
    np.savez(
        path, boxes=boxes, valid=np.ones(FRAME_COUNT, dtype=bool),
        frame=np.arange(FRAME_COUNT),
    )


def _valid_camera(frame_count=FRAME_COUNT):
    rotations = np.tile(np.eye(3, dtype=np.float32), (frame_count, 1, 1))
    return {
        'pred_cam_R': rotations.copy(),
        'pred_cam_T': np.zeros((frame_count, 3), dtype=np.float32),
        'world_cam_R': rotations.copy(),
        'world_cam_T': np.zeros((frame_count, 3), dtype=np.float32),
        'img_focal': np.asarray(CALIB[0]),
        'img_center': CALIB[2:].copy(),
        'spec_focal': np.asarray(9.0),
    }


class _FakeCapture:
    def __init__(self, fps=24.0):
        self.fps = fps
        self.index = 0

    def isOpened(self):
        return True

    def get(self, _property):
        return self.fps

    def read(self):
        if self.index == FRAME_COUNT:
            return False, None
        image = np.full((*FRAME_SHAPE, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, image

    def release(self):
        pass


def _fake_frontend(calls):
    class FakeModel:
        def cuda(self):
            return self

        def eval(self):
            return self

    class FakePredictor:
        def __init__(self, model):
            self.shape = None

        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, box, multimask_output):
            calls['sam'] += 1
            return np.ones((1, *self.shape), dtype=bool), None, None

    segment_anything = types.ModuleType('segment_anything')
    segment_anything.SamPredictor = FakePredictor
    segment_anything.sam_model_registry = {'vit_h': lambda checkpoint: FakeModel()}

    lib_camera = types.ModuleType('lib.camera')

    def calibrate(_frames, _masks, is_static=False):
        calls['calibrate'] += 1
        return CALIB.copy(), is_static

    def run_slam(_frames, masks, calib, is_static=False):
        calls['slam'] += 1
        rotations = torch.eye(3).repeat(FRAME_COUNT, 1, 1)
        return rotations, torch.zeros(FRAME_COUNT, 3)

    def align(_image, rotation, translation):
        calls['align'] += 1
        return rotation, translation, np.asarray(9.0)

    lib_camera.calibrate_intrinsics = calibrate
    lib_camera.run_metric_slam = run_slam
    lib_camera.align_cam_to_world = align
    return segment_anything, lib_camera


def _run_main(video, track, output, mode='moving'):
    argv = [
        'prepare_robot_video_camera.py', '--video', str(video), '--boxes', str(track),
        '--output-dir', str(output), '--camera-mode', mode,
    ]
    with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
        camera_script.main()


def _hash_with_fake_checkpoints(path):
    path = Path(path)
    if path.parent == ROOT / 'data/pretrain':
        return 'synthetic-' + path.name
    return REAL_SHA256(path)


REAL_SHA256 = camera_script.sha256_file


class CameraCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'synthetic-video-a')
        self.track = self.root / 'track.npz'
        _write_track(self.track)

    def tearDown(self):
        self.temporary.cleanup()

    def _initialize(self, output, mode='moving', track=None):
        output.mkdir(parents=True, exist_ok=True)
        identity = camera_script.make_cache_identity(self.video, track or self.track, mode)
        camera_script.prepare_output_cache(output, identity)
        return identity

    def _write_masks_and_intrinsics(self, output, mode='moving'):
        np.save(
            output / 'human_masks.npy',
            np.ones((FRAME_COUNT, *FRAME_SHAPE), dtype=bool),
        )
        np.savez(
            output / 'intrinsics.npz', calib=CALIB, detected_static=False, mode=mode,
        )

    def test_first_run_and_same_input_complete_cache_reuse(self):
        output = self.root / 'output'
        calls = {'sam': 0, 'calibrate': 0, 'slam': 0, 'align': 0}
        segment_anything, lib_camera = _fake_frontend(calls)
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture()), \
                mock.patch.dict(sys.modules, {
                    'segment_anything': segment_anything, 'lib.camera': lib_camera,
                }), mock.patch.object(camera_script, 'sha256_file', side_effect=_hash_with_fake_checkpoints):
            _run_main(self.video, self.track, output)
        self.assertEqual(calls, {'sam': 32, 'calibrate': 1, 'slam': 1, 'align': 1})
        self.assertTrue((output / 'cache_identity.json').is_file())
        validated = camera_script.load_camera(output / 'camera.npy', FRAME_COUNT, CALIB)
        self.assertEqual(validated['world_cam_R'].shape, (FRAME_COUNT, 3, 3))
        with np.load(output / 'boxes.npz', allow_pickle=False) as boxes:
            self.assertEqual(boxes['boxes'].shape, (FRAME_COUNT, 4))
            self.assertEqual(float(boxes['img_focal']), float(CALIB[0]))

        # No frontend modules or checkpoint hashes are needed for a complete rerun.
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture()):
            _run_main(self.video, self.track, output)
        self.assertEqual(calls['slam'], 1)

    def test_interrupted_cache_reuses_masks_and_intrinsics(self):
        output = self.root / 'output'
        self._initialize(output)
        self._write_masks_and_intrinsics(output)
        calls = {'sam': 0, 'calibrate': 0, 'slam': 0, 'align': 0}
        _segment_anything, lib_camera = _fake_frontend(calls)

        def unexpected_calibration(*args, **kwargs):
            raise AssertionError('verified intrinsics must be reused')

        lib_camera.calibrate_intrinsics = unexpected_calibration
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture()), \
                mock.patch.dict(sys.modules, {'lib.camera': lib_camera}), \
                mock.patch.object(camera_script, 'sha256_file', side_effect=_hash_with_fake_checkpoints):
            _run_main(self.video, self.track, output)
        self.assertEqual(calls['slam'], 1)
        self.assertEqual(calls['align'], 1)
        self.assertTrue((output / 'boxes.npz').is_file())

    def test_mismatched_track_mode_and_legacy_cache_fail_before_video_work(self):
        output = self.root / 'output'
        self._initialize(output)
        other_track = self.root / 'other.npz'
        _write_track(other_track, offset=1.0)
        other_video = self.root / 'other.mp4'
        other_video.write_bytes(b'synthetic-video-b')
        with mock.patch.object(
            camera_script.cv2, 'VideoCapture', side_effect=AssertionError('video work must not start')
        ):
            with self.assertRaisesRegex(ValueError, 'use a new output directory'):
                _run_main(other_video, self.track, output)
            with self.assertRaisesRegex(ValueError, 'use a new output directory'):
                _run_main(self.video, other_track, output)
            with self.assertRaisesRegex(ValueError, 'use a new output directory'):
                _run_main(self.video, self.track, output, mode='static')

        legacy = self.root / 'legacy'
        legacy.mkdir()
        np.save(legacy / 'human_masks.npy', np.ones((1, 1), dtype=bool))
        with mock.patch.object(
            camera_script.cv2, 'VideoCapture', side_effect=AssertionError('video work must not start')
        ):
            with self.assertRaisesRegex(ValueError, 'use a new output directory'):
                _run_main(self.video, self.track, legacy)

    def test_malformed_cached_camera_is_not_published(self):
        output = self.root / 'output'
        self._initialize(output)
        self._write_masks_and_intrinsics(output)
        malformed = _valid_camera()
        malformed['world_cam_R'][0] = 2 * np.eye(3)
        np.save(output / 'camera.npy', malformed)
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture()):
            with self.assertRaisesRegex(ValueError, 'proper rotation'):
                _run_main(self.video, self.track, output)
        self.assertFalse((output / 'boxes.npz').exists())

    def test_malformed_shapes_rotations_and_intrinsics_are_rejected(self):
        with self.subTest('mask count'):
            with self.assertRaisesRegex(ValueError, 'human_masks.npy must have shape'):
                camera_script.validate_masks(
                    np.ones((FRAME_COUNT - 1, *FRAME_SHAPE), dtype=bool), FRAME_COUNT, FRAME_SHAPE,
                )
        with self.subTest('camera shape'):
            malformed = _valid_camera()
            malformed['pred_cam_R'] = malformed['pred_cam_R'][:-1]
            with self.assertRaisesRegex(ValueError, 'pred_cam_R must have shape'):
                camera_script.validate_camera(malformed, FRAME_COUNT, CALIB)
        with self.subTest('rotation'):
            malformed = _valid_camera()
            malformed['pred_cam_R'][0, 0, 1] = 1.0
            with self.assertRaisesRegex(ValueError, 'proper rotation'):
                camera_script.validate_camera(malformed, FRAME_COUNT, CALIB)
        with self.subTest('intrinsic shape'):
            with self.assertRaisesRegex(ValueError, 'calib must be finite'):
                camera_script.validate_intrinsics(
                    CALIB[:3], np.asarray(False), np.asarray('moving'), 'moving',
                )
        with self.subTest('intrinsic values'):
            invalid = CALIB.copy()
            invalid[0] = np.nan
            with self.assertRaisesRegex(ValueError, 'calib must be finite'):
                camera_script.validate_intrinsics(
                    invalid, np.asarray(False), np.asarray('moving'), 'moving',
                )
        with self.subTest('intrinsic mode'):
            with self.assertRaisesRegex(ValueError, 'does not match requested mode'):
                camera_script.validate_intrinsics(
                    CALIB, np.asarray(False), np.asarray('static'), 'moving',
                )

    def test_invalid_fps_and_frame_count_are_explicit_errors(self):
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture(fps=0)):
            with self.assertRaisesRegex(ValueError, 'invalid FPS'):
                camera_script.extract_frames(self.video, self.root / 'zero-fps', FRAME_COUNT)
        with mock.patch.object(camera_script.cv2, 'VideoCapture', return_value=_FakeCapture()):
            with self.assertRaisesRegex(ValueError, 'Video/track frame mismatch'):
                camera_script.extract_frames(self.video, self.root / 'wrong-count', FRAME_COUNT + 1)


if __name__ == '__main__':
    unittest.main()
