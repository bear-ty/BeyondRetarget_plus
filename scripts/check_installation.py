"""Validate source files, robot descriptions, and optional runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PLUS_BUNDLE_SIZE = 2_875_791_008
PLUS_BUNDLE_SHA256 = "0906435b395766123b2d78364f7d8860940fbc27a09c847fbe010ae3d524ed2b"
PLUS_ROBOTS = {
    "g1",
    "r1",
    "gr1t1_without_hand",
    "gr2v3_8_7_dummy_hand",
    "h1_with_hand_without_hand",
    "t1_serial",
    "tienkung",
    "atlas_v4",
}
SUPPORTED_ROBOTS = {
    "g1": "configs/robot_g1.yaml",
    "r1": "configs/robot_r1.yaml",
    "gr1t1_without_hand": "configs/robot_gr1t1_without_hand.yaml",
    "gr2v3_8_7_dummy_hand": "configs/robot_gr2v3_8_7_dummy_hand.yaml",
    "h1_with_hand_without_hand": "configs/robot_h1_with_hand_without_hand.yaml",
    "t1_serial": "configs/robot_t1_serial.yaml",
    "tienkung": "configs/robot_tienkung.yaml",
    "atlas_v4": "configs/robot_atlas_v4.yaml",
}

# Import names differ from the names used by pip.  These are the packages
# needed by the core robot-adapter, training, and post-processing paths.
CORE_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pytorch3d": "pytorch3d",
    "smplx": "smplx",
    "chumpy": "chumpy",
    "joblib": "joblib",
    "ultralytics": "ultralytics",
    "einops": "einops",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "mujoco": "mujoco",
    "OpenGL": "PyOpenGL",
    "timm": "timm",
    "tensorboard": "tensorboard",
    "imageio_ffmpeg": "imageio[ffmpeg]",
    "tqdm": "tqdm",
}

FRONTEND_MODULES = {
    "segment_anything": "segment-anything",
    "open3d": "open3d",
    "pyrender": "pyrender",
    "gdown": "gdown",
    "pandas": "pandas",
    "omegaconf": "omegaconf",
    "torchmin": "pytorch-minimize",
    "ffmpeg": "ffmpeg-python",
    "matplotlib": "matplotlib",
    "trimesh": "trimesh",
    "plyfile": "plyfile",
    "evo": "evo",
    "pycocotools": "pycocotools",
}

EVALUATION_MODULES = {
    "openpyxl": "openpyxl",
}

EXPECTED_VERSIONS = {
    "torch": "2.3.0",
    "torchvision": "0.18.0",
    "numpy": "1.23.5",
    "opencv-python": "4.11.0.86",
    "scikit-image": "0.24.0",
    "scipy": "1.15.3",
    "chumpy": "0.70",
    "PyOpenGL": "3.1.10",
}


def check_file(path: Path, label: str, required: bool = True) -> bool:
    exists = path.is_file() and path.stat().st_size > 0
    state = "OK" if exists else ("MISSING" if required else "optional")
    print(f"[{state}] {label}: {path}")
    return exists or not required


def check_path(path: Path, label: str, required: bool = True) -> bool:
    exists = path.exists()
    state = "OK" if exists else ("MISSING" if required else "optional")
    print(f"[{state}] {label}: {path}")
    return exists or not required


def check_module(module_name: str, package: str, category: str = "Python module") -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(f"[MISSING] {category}: {module_name} ({package}): {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {category}: {module_name} ({package})")
    return True


def _version_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(expected + "+")


def check_distribution_version(distribution: str, expected: str, runtime_module: str | None = None) -> bool:
    ok = True
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        print(f"[MISSING] Python distribution: {distribution} (expected {expected})")
        return False
    if _version_matches(installed, expected):
        print(f"[OK] Python distribution: {distribution}=={installed}")
    else:
        print(f"[MISSING] Python distribution: {distribution}=={installed}; expected {expected}")
        ok = False
    if runtime_module is not None:
        try:
            module = importlib.import_module(runtime_module)
            runtime_version = str(module.__version__)
        except Exception as exc:
            print(f"[MISSING] Runtime version for {distribution}: {type(exc).__name__}: {exc}")
            return False
        if runtime_version != installed:
            location = getattr(module, "__file__", "<unknown>")
            print(
                f"[MISSING] {distribution} distribution/runtime mismatch: "
                f"{installed} != {runtime_version} ({location})"
            )
            ok = False
    return ok


def resolve_robot_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("assets", "robot"):
        robot_root = Path(os.environ.get("BRPLUS_ROBOT_ROOT", str(ROOT / "assets" / "robot")))
        return robot_root.joinpath(*path.parts[2:])
    return ROOT / path


def _resolve_mesh_paths(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    if root.tag == "robot":
        compiler = root.find("mujoco/compiler")
        mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            path = Path(filename)
            yield path if path.is_absolute() else xml_path.parent / mesh_dir / path
        return
    if root.tag == "mujoco":
        compiler = root.find("compiler")
        mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
        assets = root.find("asset")
        if assets is None:
            return
        for mesh in assets.findall("mesh"):
            filename = mesh.attrib.get("file")
            if not filename:
                continue
            path = Path(filename)
            yield path if path.is_absolute() else xml_path.parent / mesh_dir / path


def check_robot_description(spec: dict) -> bool:
    name = str(spec.get("name", "robot"))
    xml_path = resolve_robot_path(str(spec["xml_path"]))
    ok = check_file(xml_path, f"{name} robot description")
    if xml_path.is_file():
        try:
            mesh_paths = list(_resolve_mesh_paths(xml_path))
        except Exception as exc:
            print(f"[MISSING] {name} robot XML mesh validation: {exc}")
            ok = False
        else:
            missing = [path for path in mesh_paths if not path.is_file()]
            if missing:
                suffix = f" (+ {len(missing) - 1} more)" if len(missing) > 1 else ""
                print(f"[MISSING] {name} robot mesh: {missing[0]}{suffix}")
                ok = False
            else:
                print(f"[OK] {name} robot meshes: {len(mesh_paths)} referenced files")
    for side, item in dict(spec.get("foot_grounding_meshes", {})).items():
        mesh_path = resolve_robot_path(str(item["mesh_path"]))
        ok &= check_file(mesh_path, f"{name} {side} foot grounding mesh")
    return ok


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_plus_bundle(path: Path, required: bool) -> bool:
    if not path.is_file():
        print(f"[{'MISSING' if required else 'optional'}] Plus eight-robot bundle: {path}")
        return not required
    ok = True
    if path.stat().st_size != PLUS_BUNDLE_SIZE:
        print(f"[MISSING] Plus eight-robot bundle has size {path.stat().st_size}; expected {PLUS_BUNDLE_SIZE}")
        return False
    actual = sha256(path)
    if actual != PLUS_BUNDLE_SHA256:
        print(f"[MISSING] Plus eight-robot bundle SHA256 mismatch: {actual} != {PLUS_BUNDLE_SHA256}")
        return False
    print(f"[OK] Plus eight-robot bundle: complete release SHA256 verified ({path})")
    if not required:
        return True
    try:
        import torch

        payload = torch.load(path, map_location="meta", weights_only=False)
        if not isinstance(payload, dict) or len(payload.get("human", {})) != 667:
            raise ValueError("embedded TRAM/VIMO backbone must contain 667 state entries")
        if set(payload.get("robots", {})) != PLUS_ROBOTS:
            raise ValueError(f"robot heads differ from release set: {sorted(payload.get('robots', {}))}")
        if len(payload.get("static_head", {}).get("model", {})) != 55:
            raise ValueError("embedded camera-conditioned static head is incomplete")
        if payload.get("bundle_format") != "beyondretarget_plus_decoder6_bundle_v2":
            raise ValueError(f"unsupported bundle format: {payload.get('bundle_format')!r}")
    except Exception as exc:
        print(f"[MISSING] Plus eight-robot bundle structure: {exc}")
        return False
    print("[OK] Plus eight-robot bundle structure: TRAM/VIMO + eight robot heads + static head")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("TRAM_DATA_ROOT", str(ROOT / "data" / "datasets")))
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument(
        "--require-core-checkpoints",
        action="store_true",
        help="Require the complete Plus bundle and YOLO detection/pose weights used by core inference.",
    )
    parser.add_argument(
        "--require-frontend",
        action="store_true",
        help="Also require optional TRAM camera/frontend Python packages.",
    )
    parser.add_argument(
        "--require-evaluation",
        action="store_true",
        help="Also require optional evaluation/export Python packages.",
    )
    parser.add_argument(
        "--require-frontend-checkpoints",
        action="store_true",
        help="Require DROID, SAM, DEVA, and camcalib weights used by the optional camera frontend.",
    )
    parser.add_argument("--require-smpl", action="store_true")
    args = parser.parse_args()

    ok = True
    for relative in ("train.py", "scripts/train_beyondretarget.py", "scripts/infer_beyondretarget.py", "scripts/postprocess_beyondretarget.py"):
        ok &= check_file(ROOT / relative, relative)
    for robot, config in SUPPORTED_ROBOTS.items():
        ok &= check_file(ROOT / config, f"{robot} config")
    try:
        import yaml
        for config in SUPPORTED_ROBOTS.values():
            spec = yaml.safe_load((ROOT / config).read_text())
            ok &= check_robot_description(spec)
    except Exception as exc:
        print(f"[MISSING] YAML validation: {exc}")
        ok = False

    data_root = Path(args.data_root)
    check_path(data_root / "dataset_ann", "dataset annotation directory", required=False)
    smpl_required = args.require_smpl
    for name in ("SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"):
        ok &= check_file(ROOT / "data" / "smpl" / name, f"{name}", required=smpl_required)
    checkpoints = {
        "Plus eight-robot bundle": ROOT / "assets/checkpoints/beyondretarget_plus_8robots.pth",
        "HMR2b": ROOT / "data/pretrain/hmr2b/epoch=35-step=1000000.ckpt",
        "VIMO": ROOT / "data/pretrain/vimo_checkpoint.pth.tar",
        "DROID": ROOT / "data/pretrain/droid.pth",
        "SAM": ROOT / "data/pretrain/sam_vit_h_4b8939.pth",
        "DEVA": ROOT / "data/pretrain/DEVA-propagation.pth",
        "YOLO": ROOT / "assets/yolo/yolov8x.pt",
        "YOLO pose": ROOT / "assets/yolo/yolov8x-pose.pt",
        "camcalib": ROOT / "data/pretrain/camcalib_sa_biased_l2.ckpt",
    }
    # The release bundle embeds the frozen VIMO/TRAM human backbone.  A
    # standalone VIMO file is only needed for feature-cache generation and
    # training, not for the published inference path.
    required_checkpoints = {
        "YOLO", "YOLO pose", "Plus eight-robot bundle",
    } if args.require_core_checkpoints else set()
    if args.require_frontend_checkpoints:
        required_checkpoints.update({"DROID", "SAM", "DEVA", "camcalib"})
    if args.require_checkpoints:
        required_checkpoints.update(checkpoints.keys())
    for label, path in checkpoints.items():
        if label == "Plus eight-robot bundle":
            ok &= check_plus_bundle(path, required=label in required_checkpoints)
        else:
            ok &= check_file(path, label, required=label in required_checkpoints)
    for module, package in CORE_MODULES.items():
        ok &= check_module(module, package)
    for distribution, expected in EXPECTED_VERSIONS.items():
        runtime_module = "numpy" if distribution == "numpy" else None
        ok &= check_distribution_version(distribution, expected, runtime_module=runtime_module)
    if args.require_frontend:
        for module, package in FRONTEND_MODULES.items():
            ok &= check_module(module, package, category="Frontend module")
    if args.require_evaluation:
        for module, package in EVALUATION_MODULES.items():
            ok &= check_module(module, package, category="Evaluation module")
    print("Installation check passed." if ok else "Installation check found required items that are missing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
