"""Drop-in component integration with original minimal fixtures, without upstream code."""

import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brplus_external_checker', ROOT / 'scripts/check_installation.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ExternalComponentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in ('lib/__init__.py', 'lib/datasets/__init__.py',
                         'lib/datasets/coco_occlusion.py',
                         'lib/util/beyondretarget/external_components.py'):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def _write(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')

    def _run(self, code):
        return subprocess.run(
            [sys.executable, '-I', '-S', '-B', '-c',
             f'import sys; sys.path.insert(0, {str(self.root)!r})\n' + code],
            capture_output=True, text=True, check=False, timeout=15,
        )

    def _camera_source(self):
        self._write('thirdparty/camcalib/__init__.py', '')
        self._write('thirdparty/camcalib/cam_utils.py', 'TOKEN = object()\n')
        self._write('thirdparty/camcalib/resnet.py', 'VALUE = 123\n')
        self._write('thirdparty/camcalib/model.py',
                    'from .cam_utils import TOKEN\nfrom .resnet import VALUE\n'
                    'class CameraRegressorNetwork:\n'
                    '    token = TOKEN\n    value = VALUE\n'
                    '    def __init__(self):\n'
                    '        raise AssertionError("readiness must not construct a model")\n'
                    '    def load_ckpt(self, path):\n        return self\n')

    def test_default_imports_do_not_load_optional_sources(self):
        result = self._run(
            'from lib.util.beyondretarget import external_components\n'
            'from lib.datasets.coco_occlusion import occlude_with_pascal_objects\n'
            'assert "thirdparty.camcalib.model" not in sys.modules\n'
            'assert "thirdparty.pare.coco_occlusion" not in sys.modules\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_and_partial_camera_sources_explain_fixed_placement(self):
        code = ('from lib.util.beyondretarget.external_components import load_camera_regressor\n'
                'load_camera_regressor()\n')
        for partial in (False, True):
            with self.subTest(partial=partial):
                if partial:
                    self._write('thirdparty/camcalib/model.py', 'raise AssertionError("incomplete source")\n')
                result = self._run(code)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FileNotFoundError', result.stderr)
                self.assertIn(str(self.root / 'thirdparty/camcalib'), result.stderr)
                self.assertIn('resnet.py', result.stderr)
                self.assertIn('README.md#spec-and-pare-components', result.stderr)
                self.assertNotIn('AssertionError', result.stderr)

    def test_camera_relative_imports_and_nonempty_checkpoint_without_model_creation(self):
        self._camera_source()
        code = ('from lib.util.beyondretarget.external_components import require_camera_resources\n'
                'model, checkpoint = require_camera_resources()\n'
                'assert model.value == 123\n'
                'assert checkpoint.name == "camcalib_sa_biased_l2.ckpt"\n')
        checkpoint = 'data/pretrain/camcalib_sa_biased_l2.ckpt'
        for content in (None, '', 'synthetic-checkpoint'):
            with self.subTest(content=content):
                if content is not None:
                    self._write(checkpoint, content)
                result = self._run(code)
                if content:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertIn('Missing SPEC/camcalib checkpoint', result.stderr)
                    self.assertIn(str(self.root / checkpoint), result.stderr)

    def test_camera_wrong_api_and_transitive_import_errors_remain_visible(self):
        self._camera_source()
        code = ('from lib.util.beyondretarget.external_components import load_camera_regressor\n'
                'load_camera_regressor()\n')
        for source, error in (
            ('class CameraRegressorNetwork: pass\n', 'CameraRegressorNetwork.load_ckpt API'),
            ('import missing_brplus_camera_dependency\n', 'ModuleNotFoundError'),
            ('raise ImportError("synthetic ABI failure")\n', 'synthetic ABI failure'),
        ):
            with self.subTest(error=error):
                self._write('thirdparty/camcalib/model.py', source)
                result = self._run(code)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('source is not included', result.stderr)

    def test_pascal_shim_uses_fixed_path_and_preserves_arguments_and_exceptions(self):
        code = ('from lib.datasets.coco_occlusion import occlude_with_pascal_objects\n'
                'image, occluders = object(), object()\n'
                'result = occlude_with_pascal_objects(image, occluders)\n'
                'assert result[0] is image and result[1] is occluders\n')
        missing = self._run(code)
        self.assertIn('FileNotFoundError', missing.stderr)
        self.assertIn(str(self.root / 'thirdparty/pare/coco_occlusion.py'), missing.stderr)
        self._write('thirdparty/pare/coco_occlusion.py',
                    'def occlude_with_pascal_objects(im, occluders):\n    return im, occluders\n')
        supplied = self._run(code)
        self.assertEqual(supplied.returncode, 0, supplied.stderr)
        self._write('thirdparty/pare/coco_occlusion.py',
                    'def occlude_with_pascal_objects(im, occluders):\n'
                    '    raise ValueError("upstream augmentation failed")\n')
        failed = self._run(code)
        self.assertIn('ValueError: upstream augmentation failed', failed.stderr)

    def test_checker_probes_supplied_sources_and_occluder_data(self):
        output = io.StringIO()
        with mock.patch.object(checker, 'ROOT', self.root), contextlib.redirect_stdout(output):
            self.assertFalse(checker.check_external_source('camcalib'))
            self.assertFalse(checker.check_occlusion_resources())
            self._camera_source()
            self.assertTrue(checker.check_external_source('camcalib'))
            self._write('thirdparty/pare/coco_occlusion.py', 'VALUE = 1\n')
            self._write('data/pascal_occluders.pkl', 'synthetic-resource')
            self.assertFalse(checker.check_occlusion_resources())
            self._write('thirdparty/pare/coco_occlusion.py',
                        'def occlude_with_pascal_objects(im, occluders):\n    return im\n')
            self._write('data/pascal_occluders.pkl', '')
            self.assertFalse(checker.check_occlusion_resources())
            self._write('data/pascal_occluders.pkl', 'synthetic-resource')
            self.assertTrue(checker.check_occlusion_resources())
        self.assertIn('README.md#spec-and-pare-components', output.getvalue())

    def test_git_ignores_user_sources_but_keeps_component_instructions(self):
        if not (ROOT / '.git').exists():
            self.skipTest('Git checkout required')
        for relative, ignored in (
            ('thirdparty/camcalib/model.py', True),
            ('thirdparty/camcalib/__init__.py', True),
            ('thirdparty/camcalib/nested/copied.py', True),
            ('thirdparty/pare/coco_occlusion.py', True),
            ('thirdparty/pare/nested/copied.py', True),
            ('thirdparty/camcalib/README.md', False),
            ('thirdparty/pare/README.md', False),
        ):
            with self.subTest(path=relative):
                result = subprocess.run(
                    ['git', 'check-ignore', '--no-index', '--verbose', '-z', '--stdin'],
                    input=relative + '\0', cwd=ROOT, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                pattern = result.stdout.split('\0')[2]
                self.assertEqual(not pattern.startswith('!'), ignored, result.stdout)


if __name__ == '__main__':
    unittest.main()
