"""Standard-library-only offline tests for installation and asset regressions."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_project_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


downloader = load_project_module("brplus_offline_downloader", "scripts/download_base_shared_assets.py")
checker = load_project_module("brplus_offline_checker", "scripts/check_installation.py")


class RequirementAndScriptTests(unittest.TestCase):
    def test_dependency_and_distribution_mapping(self):
        requirements = [
            line.strip()
            for line in (ROOT / "requirements_beyondretarget.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        expected = {
            "numpy==1.23.5",
            "opencv-python==4.11.0.86",
            "scikit-image==0.24.0",
            "scipy==1.15.3",
            "chumpy==0.70",
            "joblib",
            "openpyxl",
            "pytorch-minimize",
            "PyOpenGL==3.1.10",
            "pyrender @ git+https://github.com/mmatl/pyrender.git@7c613e8aed7142df9ff40767a8f10b7a19b6255c",
        }
        self.assertTrue(expected.issubset(requirements))
        self.assertNotIn("torchmin", requirements)
        self.assertEqual(checker.FRONTEND_MODULES["torchmin"], "pytorch-minimize")
        self.assertEqual(checker.CORE_MODULES["OpenGL"], "PyOpenGL")
        self.assertEqual(checker.EVALUATION_MODULES["openpyxl"], "openpyxl")

    def test_setup_checks_numpy_distribution_and_runtime(self):
        setup = (ROOT / "scripts/setup_environment.sh").read_text(encoding="utf-8")
        self.assertIn('expected = "1.23.5"', setup)
        self.assertIn('distribution_version = metadata.version("numpy")', setup)
        self.assertIn("runtime_version = numpy.__version__", setup)
        self.assertIn("do not delete numpy/_core manually", setup)

    def test_pascal_downloader_uses_portable_python_default(self):
        script = (ROOT / "scripts/download_pascal_occluders.sh").read_text(encoding="utf-8")
        self.assertIn('TRAM_PYTHON="${TRAM_PYTHON:-python}"', script)
        self.assertNotIn("/nasdata/tianyu", script)


class RobotArchiveRegressionTests(unittest.TestCase):
    def test_provider_aliases_fill_missing_meshes_and_preserve_canonical_xml(self):
        members = {
            "robot/unitree_g1/g1_mocap_29dof.xml": b"provider-g1-xml",
            "robot/unitree_g1/meshes/pelvis.STL": b"g1-pelvis",
            "robot/unitree_g1/meshes/nested/hip.STL": b"g1-hip",
            "robot/unitree_r1/r1_mocap.xml": b"provider-r1-xml",
            "robot/unitree_r1/meshes/left_ankle_roll_link.STL": b"r1-ankle",
            "robot/gr1t1/gr1t1_fourier_hand_6dof.urdf": b"gr1",
            "robot/gr2v3_8_7_dummy_hand/gr2v3_8_7_dummy_hand.urdf": b"gr2",
            "robot/h1_with_hand/h1_with_hand.urdf": b"provider-h1-xml",
            "robot/h1_with_hand/meshes/pelvis.STL": b"h1-pelvis",
            "robot/t1_serial/T1_serial.urdf": b"t1",
            "robot/tienkung/tienkung2_lite.urdf": b"tienkung",
            "robot/atlas_v4/atlas_v4_with_multisense.urdf": b"atlas",
        }
        canonical = {
            "g1/g1_mocap_29dof.xml": b"canonical-g1-xml",
            "r1/r1.xml": b"canonical-r1-xml",
            "h1_with_hand_without_hand/h1_with_hand_without_hand.urdf": b"canonical-h1-xml",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            robot_root = assets / "robot"
            robot_root.mkdir(parents=True)
            with zipfile.ZipFile(assets / "robot.zip", "w") as archive:
                for name, payload in members.items():
                    archive.writestr(name, payload)
            for relative, payload in canonical.items():
                path = robot_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            with mock.patch.object(downloader, "ROOT", root), mock.patch.object(
                downloader, "download_file", side_effect=AssertionError("network access is forbidden")
            ):
                downloader.ensure_robot({"id": "offline"}, force=False)
                # A second pass exercises the already-materialized archive path.
                downloader.ensure_robot({"id": "offline"}, force=False)

            for relative, payload in canonical.items():
                self.assertEqual((robot_root / relative).read_bytes(), payload)
            self.assertEqual((robot_root / "g1/meshes/pelvis.STL").read_bytes(), b"g1-pelvis")
            self.assertEqual((robot_root / "g1/meshes/nested/hip.STL").read_bytes(), b"g1-hip")
            self.assertEqual(
                (robot_root / "r1/meshes/left_ankle_roll_link.STL").read_bytes(), b"r1-ankle"
            )
            self.assertEqual((robot_root / "h1_with_hand_without_hand/meshes/pelvis.STL").read_bytes(), b"h1-pelvis")


class InstallationCheckerTests(unittest.TestCase):
    def test_module_check_propagates_real_import_failure(self):
        with mock.patch.object(checker.importlib, "import_module", side_effect=ImportError("broken ABI")) as imported:
            with contextlib.redirect_stdout(io.StringIO()):
                available = checker.check_module("numpy", "numpy")
        self.assertFalse(available)
        imported.assert_called_once_with("numpy")

    def test_numpy_distribution_runtime_mismatch_fails(self):
        runtime = SimpleNamespace(__version__="2.2.6", __file__="/mixed/numpy/__init__.py")
        with mock.patch.object(checker.metadata, "version", return_value="1.23.5"), mock.patch.object(
            checker.importlib, "import_module", return_value=runtime
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                valid = checker.check_distribution_version("numpy", "1.23.5", runtime_module="numpy")
        self.assertFalse(valid)

    def test_mjcf_mesh_check_fails_then_passes_when_mesh_is_materialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot_root = Path(temporary) / "robot"
            xml_directory = robot_root / "g1"
            mesh_directory = xml_directory / "meshes"
            mesh_directory.mkdir(parents=True)
            xml_path = xml_directory / "g1.xml"
            xml_path.write_text(
                "<mujoco><compiler meshdir='meshes'/><asset>"
                "<mesh name='a' file='a.STL'/><mesh name='b' file='nested/b.STL'/>"
                "</asset></mujoco>",
                encoding="utf-8",
            )
            (mesh_directory / "a.STL").write_bytes(b"a")
            spec = {
                "name": "g1",
                "xml_path": "assets/robot/g1/g1.xml",
                "foot_grounding_meshes": {
                    "left_foot": {"mesh_path": "assets/robot/g1/meshes/a.STL"},
                },
            }
            with mock.patch.dict(os.environ, {"BRPLUS_ROBOT_ROOT": str(robot_root)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(checker.check_robot_description(spec))
                (mesh_directory / "nested").mkdir()
                (mesh_directory / "nested/b.STL").write_bytes(b"b")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(checker.check_robot_description(spec))

    def test_urdf_mesh_paths_are_relative_to_the_description(self):
        with tempfile.TemporaryDirectory() as temporary:
            xml_path = Path(temporary) / "r1.urdf"
            xml_path.write_text(
                "<robot><link name='a'><visual><geometry>"
                "<mesh filename='meshes/a.STL'/><mesh filename='package://ignored.STL'/>"
                "</geometry></visual></link></robot>",
                encoding="utf-8",
            )
            self.assertEqual(
                list(checker._resolve_mesh_paths(xml_path)),
                [xml_path.parent / "meshes/a.STL"],
            )


if __name__ == "__main__":
    unittest.main()
