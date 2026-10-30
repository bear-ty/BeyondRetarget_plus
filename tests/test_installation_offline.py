"""Standard-library-only offline tests for installation and asset regressions."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
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

    def test_conda_activation_relaxes_only_nounset_and_propagates_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conda_root = root / "conda"
            activation = conda_root / "etc/conda/activate.d"
            activation.mkdir(parents=True)
            profile = conda_root / "etc/profile.d"
            profile.mkdir(parents=True)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            log = root / "calls.log"
            (activation / "mkl.sh").write_text(
                'printf "hook:mkl=%s\\n" "${MKL_INTERFACE_LAYER-unset}" >> "$FAKE_CONDA_LOG"\n'
                'if [[ "${FAKE_CONDA_HOOK_COMMAND_FAIL:-0}" == 1 ]]; then false; fi\n'
                ': "${MKL_INTERFACE_LAYER}"\n',
                encoding="utf-8",
            )
            (profile / "conda.sh").write_text(
                '''if [[ "${FAKE_CONDA_SOURCE_COMMAND_FAIL:-0}" == 1 ]]; then false; fi
conda() {
  local state=without-u
  [[ $- == *u* ]] && state=with-u
  printf 'conda:%s:%s:%s\\n' "$state" "$1" "${2:-}" >> "$FAKE_CONDA_LOG"
  case "$1" in
    env) printf 'BeyondRetarget_plus\\n'; return 0 ;;
    activate)
      source "$FAKE_CONDA_ROOT/etc/conda/activate.d/mkl.sh"
      return "${FAKE_CONDA_ACTIVATE_STATUS:-0}" ;;
  esac
  return 99
}
python() {
  local state=without-u
  [[ $- == *u* ]] && state=with-u
  printf 'python:%s:%s\\n' "$state" "$*" >> "$FAKE_CONDA_LOG"
  if [[ "$1" == "-" ]]; then
    while IFS= read -r _line; do :; done
    return "${FAKE_CONDA_NUMPY_STATUS:-0}"
  fi
  return 0
}
return "${FAKE_CONDA_SOURCE_STATUS:-0}"
''',
                encoding="utf-8",
            )
            fake_conda = bin_dir / "conda"
            fake_conda.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "$1" == info && "$2" == --base ]]; then\n'
                '  printf "%s\\n" "$FAKE_CONDA_ROOT"\n'
                '  exit 0\n'
                'fi\n'
                'exit 99\n',
                encoding="utf-8",
            )
            fake_conda.chmod(0o755)

            scenarios = [
                ("success", {}, 0),
                ("source", {"FAKE_CONDA_SOURCE_STATUS": "19"}, 19),
                ("source_command", {"FAKE_CONDA_SOURCE_COMMAND_FAIL": "1"}, 1),
                ("activate", {"FAKE_CONDA_ACTIVATE_STATUS": "37"}, 37),
                ("hook_command", {"FAKE_CONDA_HOOK_COMMAND_FAIL": "1"}, 1),
                ("numpy", {"FAKE_CONDA_NUMPY_STATUS": "23"}, 23),
            ]
            for name, overrides, expected_status in scenarios:
                with self.subTest(scenario=name):
                    log.write_text("", encoding="utf-8")
                    env = os.environ.copy()
                    env.pop("MKL_INTERFACE_LAYER", None)
                    env.update({
                        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
                        "FAKE_CONDA_ROOT": str(conda_root),
                        "FAKE_CONDA_LOG": str(log),
                        "BRPLUS_DOWNLOAD_SHARED_ASSETS": "0",
                        "BRPLUS_FETCH_SUBMODULES": "0",
                        **overrides,
                    })
                    completed = subprocess.run(
                        ["bash", str(ROOT / "scripts/setup_environment.sh")],
                        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=30,
                    )
                    self.assertEqual(completed.returncode, expected_status, completed.stderr)
                    calls = log.read_text(encoding="utf-8").splitlines()
                    python_calls = [line for line in calls if line.startswith("python:")]
                    if name.startswith("source"):
                        self.assertEqual(calls, [])
                        continue
                    self.assertIn("conda:without-u:activate:BeyondRetarget_plus", calls)
                    self.assertIn("hook:mkl=unset", calls)
                    self.assertTrue(all(line.startswith("python:with-u:") for line in python_calls))
                    if name in ("activate", "hook_command"):
                        self.assertEqual(python_calls, [])
                    else:
                        self.assertTrue(any(line.startswith("python:with-u:-") for line in python_calls))
                    if name == "success":
                        self.assertIn("Environment ready: BeyondRetarget_plus", completed.stdout)

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

    def test_frontend_probe_fails_on_broken_import_not_only_missing_spec(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ImportError: broken ABI\n",
        )
        with mock.patch.object(checker.subprocess, "run", return_value=completed) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                available = checker.check_frontend_import(
                    "droid_backends", "DROID-SLAM compiled extension", paths=(checker.DROID_ROOT,),
                )
        self.assertFalse(available)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-I", "-B"])
        self.assertIn("importlib.import_module('droid_backends')", command[-1])
        self.assertLess(command[-1].index("import torch"), command[-1].index("module = importlib.import_module"))
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_strict_frontend_checks_actual_detectron2_deva_and_native_modules(self):
        with mock.patch.object(checker, "check_module", return_value=True), mock.patch.object(
            checker, "check_frontend_import", return_value=True
        ) as native, mock.patch.object(checker, "check_camera_timm", return_value=True):
            self.assertTrue(checker.check_frontend_dependencies())
        modules = {call.args[0]: call for call in native.call_args_list}
        self.assertTrue({
            "detectron2.config", "deva.model.network", "deva.inference.inference_core",
            "deva.inference.demo_utils", "droid_backends", "lietorch_backends", "lietorch",
        }.issubset(modules))
        self.assertIn(checker.DEVA_ROOT, modules["deva.model.network"].kwargs["paths"])
        self.assertIn(checker.DROID_ROOT, modules["droid_backends"].kwargs["paths"])
        self.assertIn(checker.LIETORCH_ROOT, modules["lietorch_backends"].kwargs["paths"])

    def test_camera_timm_uses_fresh_isolated_interpreter_and_documented_version(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch.dict(os.environ, {"TRAM_CAMERA_DEPS": "/camera/deps"}), mock.patch.object(
            checker.subprocess, "run", return_value=completed
        ) as run, mock.patch.object(
            checker.importlib, "import_module", side_effect=AssertionError("core timm import is forbidden")
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(checker.check_camera_timm())
        command = run.call_args.args[0]
        self.assertIn("-I", command)
        self.assertIn("expected = '0.6.7'", command[-1])
        self.assertIn("TRAM_CAMERA_DEPS", command[-1])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            checker.subprocess, "run", side_effect=AssertionError("probe should not run")
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(checker.check_camera_timm())

    def test_core_only_check_does_not_probe_optional_frontend(self):
        with mock.patch.object(checker, "SUPPORTED_ROBOTS", {}), mock.patch.object(
            checker, "CORE_MODULES", {}
        ), mock.patch.object(checker, "EXPECTED_VERSIONS", {}), mock.patch.object(
            checker, "check_file", return_value=True
        ), mock.patch.object(checker, "check_path", return_value=True), mock.patch.object(
            checker, "check_plus_bundle", return_value=True
        ), mock.patch.object(checker, "check_frontend_dependencies", return_value=True) as frontend:
            with mock.patch.object(sys, "argv", ["check_installation.py"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(checker.main(), 0)
            frontend.assert_not_called()
            with mock.patch.object(sys, "argv", ["check_installation.py", "--require-frontend"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(checker.main(), 0)
            frontend.assert_called_once_with()

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
