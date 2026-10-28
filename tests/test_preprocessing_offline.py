import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.util.beyondretarget.video_tracks import load_box_track


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coco17 = load_script('extract_video_coco17')
dataset_validator = load_script('validate_dataset')
select_pose_box = coco17.select_pose_box
validate_sequence = dataset_validator.validate_sequence


class PreprocessingCliTests(unittest.TestCase):
    def test_bedlam_help_has_no_processing_side_effects(self):
        for name in ('extract_bedlam_jpg.py', 'crop_datasets.py'):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(ROOT / 'scripts' / name), '--help'],
                    cwd=directory, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_track_requires_aligned_complete_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'track.npz'
            boxes = np.tile([0., 0., 100., 200.], (3, 1))
            np.savez(path, boxes=boxes, valid=np.ones(3, bool), frame=np.arange(3))
            track = load_box_track(path, require_complete=True)
            np.testing.assert_array_equal(track['boxes'], boxes)
            np.savez(path, boxes=boxes, valid=[True, False, True], frame=np.arange(3))
            with self.assertRaisesRegex(ValueError, 'unobserved'):
                load_box_track(path, require_complete=True)
            np.savez(path, boxes=boxes, valid=np.ones(3, bool), frame=[0, 1, 3])
            with self.assertRaisesRegex(ValueError, '0..T-1'):
                load_box_track(path)

    def test_pose_candidate_uses_iou(self):
        index, iou = select_pose_box(
            np.array([0., 0., 100., 100.]),
            np.array([[200., 200., 300., 300.], [0., 0., 100., 100.]]),
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(iou, 1.0)
        self.assertEqual(select_pose_box(np.array([0., 0., 1., 1.]), np.empty((0, 4))), (-1, 0.0))

    def test_coco17_script_writes_aligned_keypoints_with_mocked_pose_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'video.mp4'
            weights = root / 'pose.pt'
            video.touch(); weights.touch()
            boxes = np.tile([0., 0., 100., 100.], (16, 1))
            track = root / 'track.npz'
            np.savez(track, boxes=boxes, valid=np.ones(16, bool), frame=np.arange(16))
            output = root / 'coco17.npz'

            class FakeYOLO:
                def __init__(self, _weights):
                    pass

                def predict(self, **_kwargs):
                    results = []
                    for _ in range(16):
                        results.append(SimpleNamespace(
                            boxes=SimpleNamespace(xyxy=torch.tensor([[0., 0., 100., 100.]])),
                            keypoints=SimpleNamespace(
                                xy=torch.arange(34, dtype=torch.float32).reshape(1, 17, 2),
                                conf=torch.full((1, 17), 0.9),
                            ),
                        ))
                    return results

            class FakeCapture:
                def get(self, _property):
                    return 30.0

                def release(self):
                    pass

            ultralytics = ModuleType('ultralytics')
            ultralytics.YOLO = FakeYOLO
            argv = [
                'extract_video_coco17.py', '--video', str(video), '--boxes', str(track),
                '--weights', str(weights), '--output', str(output),
            ]
            with mock.patch.dict(sys.modules, {'ultralytics': ultralytics}), \
                    mock.patch.object(coco17.cv2, 'VideoCapture', return_value=FakeCapture()), \
                    mock.patch.object(sys, 'argv', argv):
                coco17.main()
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload['keypoints'].shape, (16, 17, 3))
                np.testing.assert_array_equal(payload['frame'], np.arange(16))
                self.assertEqual(float(payload['fps']), 30.0)
                self.assertTrue(payload['valid'].all())
                np.testing.assert_allclose(payload['keypoints'][0, :, 2], 0.9)

    def test_vit_features_are_opt_in_for_legacy_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory)
            np.save(sequence / 'keypoints.npy', np.zeros((2, 24, 3), dtype=np.float32))
            np.save(sequence / 'smpl.npy', {
                'transl': np.zeros((2, 3), dtype=np.float32),
                'global_orient': np.zeros((2, 3), dtype=np.float32),
                'body_pose': np.zeros((2, 69), dtype=np.float32),
                'betas': np.zeros(10, dtype=np.float32),
            })
            errors, _ = validate_sequence(sequence)
            self.assertEqual(errors, [])
            errors, _ = validate_sequence(sequence, require_vit_features=True)
            self.assertIn('missing vit_features.pt', errors)
            torch.save(torch.zeros(2, 1024), sequence / 'vit_features.pt')
            errors, _ = validate_sequence(sequence, require_vit_features=True)
            self.assertEqual(errors, [])

    def test_readme_uses_connected_inference_and_static_commands(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('scripts/extract_video_coco17.py', readme)
        self.assertIn('--boxes outputs/frontend/boxes.npz', readme)
        self.assertIn('--camera-trajectory outputs/frontend/camera.npy', readme)
        self.assertIn('torchrun --standalone --nproc_per_node=3', readme)
        self.assertIn('scripts/build_true_world_static.py', readme)
        self.assertNotIn('--boxes outputs/boxes.npz', readme)


if __name__ == '__main__':
    unittest.main()
