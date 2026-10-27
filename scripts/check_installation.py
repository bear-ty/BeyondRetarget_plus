"""Validate source files, robot descriptions, and optional runtime assets."""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path


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
    "pytorch3d": "pytorch3d",
    "smplx": "smplx",
    "ultralytics": "ultralytics",
    "einops": "einops",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "mujoco": "mujoco",
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
    "torchmin": "torchmin",
    "ffmpeg": "ffmpeg-python",
    "matplotlib": "matplotlib",
    "trimesh": "trimesh",
    "plyfile": "plyfile",
    "evo": "evo",
    "pycocotools": "pycocotools",
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
        help="Require the complete Plus bundle and YOLO weights used by core inference.",
    )
    parser.add_argument(
        "--require-frontend",
        action="store_true",
        help="Also require optional TRAM camera/frontend Python packages.",
    )
    parser.add_argument(
        "--require-frontend-checkpoints",
        action="store_true",
        help="Require DROID and SAM weights used by the optional camera frontend.",
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
            ok &= check_file(ROOT / spec["xml_path"], f"{spec['name']} robot description")
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
    }
    # The release bundle embeds the frozen VIMO/TRAM human backbone.  A
    # standalone VIMO file is only needed for feature-cache generation and
    # training, not for the published inference path.
    required_checkpoints = {
        "YOLO", "Plus eight-robot bundle",
    } if args.require_core_checkpoints else set()
    if args.require_frontend_checkpoints:
        required_checkpoints.update({"DROID", "SAM", "DEVA"})
    if args.require_checkpoints:
        required_checkpoints.update(checkpoints.keys())
    for label, path in checkpoints.items():
        if label == "Plus eight-robot bundle":
            ok &= check_plus_bundle(path, required=label in required_checkpoints)
        else:
            ok &= check_file(path, label, required=label in required_checkpoints)
    for module, package in CORE_MODULES.items():
        available = importlib.util.find_spec(module) is not None
        print(f"[{'OK' if available else 'MISSING'}] Python module: {module} ({package})")
        ok &= available
    if args.require_frontend:
        for module, package in FRONTEND_MODULES.items():
            available = importlib.util.find_spec(module) is not None
            print(f"[{'OK' if available else 'MISSING'}] Frontend module: {module} ({package})")
            ok &= available
    print("Installation check passed." if ok else "Installation check found required items that are missing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
