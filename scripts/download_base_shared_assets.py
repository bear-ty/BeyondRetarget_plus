#!/usr/bin/env python3
"""Download shared Base assets and the Plus bundle from one Drive folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = "https://drive.google.com/drive/folders/12ySWLbVhF8WEGyLR0Ll09DmZm8rZE0Tg"
YOLO_SIZE = 136_890_692
YOLO_SHA256 = "3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8"
PLUS_BUNDLE_NAMES = (
    "beyondretarget_plus_8robots.pth",
)
# The release bundle embeds the complete frozen TRAM/VIMO backbone (667 state
# entries), eight robot heads, and the shared camera-conditioned static head.
# The previous 84 MB head-only artifact is intentionally rejected.
PLUS_BUNDLE_SIZE = 2_875_791_008
PLUS_BUNDLE_SHA256 = "0906435b395766123b2d78364f7d8860940fbc27a09c847fbe010ae3d524ed2b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folder_entries(url: str) -> dict[str, dict[str, object]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BeyondRetarget-Plus"})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(.*?)';if", html, re.DOTALL)
    if not match:
        raise RuntimeError("Google Drive folder metadata was not found; use --folder-url for a public folder")
    encoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda item: chr(int(item.group(1), 16)), match.group(1))
    rows = json.loads(encoded)[0]
    result = {}
    for row in rows:
        if not isinstance(row, list) or len(row) <= 13 or not isinstance(row[0], str) or not isinstance(row[2], str):
            continue
        name = re.sub(r"\\u([0-9a-fA-F]{4})", lambda item: chr(int(item.group(1), 16)), row[2])
        result[name] = {"id": row[0], "size": int(row[13]) if isinstance(row[13], (int, float)) else None}
    return result


def download_file(file_id: str, destination: Path) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BeyondRetarget-Plus"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    os.replace(partial, destination)


def ensure_yolo(entry: dict[str, object], force: bool) -> None:
    target = ROOT / "assets/yolo/yolov8x.pt"
    if target.is_file() and not force:
        if target.stat().st_size != YOLO_SIZE or sha256(target) != YOLO_SHA256:
            raise RuntimeError("Existing yolov8x.pt failed the Base release size/SHA check; use --force")
        return
    if int(entry.get("size") or 0) != YOLO_SIZE:
        raise RuntimeError("Drive yolov8x.pt has an unexpected size")
    download_file(str(entry["id"]), target)
    if target.stat().st_size != YOLO_SIZE or sha256(target) != YOLO_SHA256:
        target.unlink(missing_ok=True)
        raise RuntimeError("Downloaded yolov8x.pt failed the Base release size/SHA check")


def safe_members(archive: zipfile.ZipFile):
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "robot":
            raise RuntimeError(f"Unsafe robot.zip member: {info.filename}")
        if mode == stat.S_IFLNK:
            raise RuntimeError(f"Symbolic links are not allowed in robot.zip: {info.filename}")
        yield info


def _materialized_robot_archive(members: list[zipfile.ZipInfo], assets_root: Path) -> bool:
    """Return whether every archive member is already materialized safely.

    A fresh Plus checkout intentionally contains only the canonical XML/URDF
    files.  That directory is not evidence that the provider meshes from
    ``robot.zip`` have been extracted, so cache validity is checked against
    every regular archive member (including its byte size).  Existing symlinks
    are rejected instead of being followed during this check or extraction.
    """
    for info in members:
        relative = PurePosixPath(info.filename)
        destination = assets_root.joinpath(*relative.parts)
        if destination.is_symlink():
            raise RuntimeError(f"Refusing symlinked robot path: {destination}")
        parent = destination.parent
        while parent != assets_root and assets_root in parent.parents:
            if parent.is_symlink():
                raise RuntimeError(f"Refusing symlinked robot directory: {parent}")
            parent = parent.parent
        if info.is_dir():
            if not destination.is_dir():
                return False
            continue
        if not destination.is_file():
            return False
        if destination.stat().st_size != info.file_size:
            return False
    return True


def ensure_robot(entry: dict[str, object], force: bool) -> None:
    archive_path = ROOT / "assets/robot.zip"
    robot_root = ROOT / "assets/robot"
    if force or not archive_path.is_file():
        download_file(str(entry["id"]), archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = list(safe_members(archive))
        names = {PurePosixPath(info.filename).parts[1] for info in members if len(PurePosixPath(info.filename).parts) > 1}
        required_groups = [
            {"g1", "unitree_g1"}, {"r1", "unitree_r1"}, {"gr1t1"},
            {"gr2v3_8_7_dummy_hand"}, {"h1_with_hand", "h1_with_hand_without_hand"},
            {"t1_serial"}, {"tienkung"}, {"atlas_v4"},
        ]
        missing = [sorted(group) for group in required_groups if not names.intersection(group)]
        if missing:
            raise RuntimeError(f"robot.zip is missing supported Plus robots: {missing}")
        if names & {"bluewhale", "penguin", "penguin_15dof", "romeo", "romeo_without_hand"}:
            raise RuntimeError("robot.zip contains an excluded robot")
        if not force and robot_root.is_dir() and _materialized_robot_archive(members, ROOT / "assets"):
            # Run alias materialization below even for a complete cache: old
            # archives may use provider directory names while configs use the
            # canonical Plus names.
            pass
        else:
            robot_root.mkdir(parents=True, exist_ok=True)
            for info in members:
                destination = ROOT / "assets" / PurePosixPath(info.filename)
                if info.is_dir():
                    if destination.exists() and destination.is_symlink():
                        raise RuntimeError(f"Refusing symlinked robot path: {destination}")
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    if destination.is_symlink():
                        raise RuntimeError(f"Refusing symlinked robot path: {destination}")
                    parent = destination.parent
                    while parent != ROOT / "assets" and (ROOT / "assets") in parent.parents:
                        if parent.is_symlink():
                            raise RuntimeError(f"Refusing symlinked robot directory: {parent}")
                        parent = parent.parent
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    # Older Base archives used provider directory names. Materialize the
    # Plus names as ordinary copies so configs never depend on symlinks.
    aliases = {
        ("unitree_g1", "g1_mocap_29dof.xml"): ("g1", "g1_mocap_29dof.xml"),
        ("unitree_r1", "r1_mocap.xml"): ("r1", "r1.xml"),
        ("gr1t1", "gr1t1_fourier_hand_6dof.urdf"): ("gr1t1", "gr1t1_without_hand.urdf"),
        ("h1_with_hand", "h1_with_hand.urdf"): ("h1_with_hand_without_hand", "h1_with_hand_without_hand.urdf"),
        ("t1_serial", "T1_serial.urdf"): ("t1_serial", "t1_serial.urdf"),
        ("tienkung", "tienkung2_lite.urdf"): ("tienkung", "tienkung.urdf"),
        ("atlas_v4", "atlas_v4_with_multisense.urdf"): ("atlas_v4", "atlas_v4.urdf"),
    }
    for (source_dir, source_file), (target_dir, target_file) in aliases.items():
        source = robot_root / source_dir
        target = robot_root / target_dir
        source_xml = source / source_file
        target_xml = target / target_file
        if source_xml.is_file() and not target_xml.is_file():
            if not target.exists():
                shutil.copytree(source, target)
            shutil.copy2(source_xml, target_xml)


def ensure_plus_bundle(entries: dict[str, dict[str, object]], force: bool) -> None:
    entry = next((entries[name] for name in PLUS_BUNDLE_NAMES if name in entries), None)
    if entry is None:
        raise RuntimeError(
            'The shared Drive folder does not contain the Plus bundle. Expected one of: '
            + ', '.join(PLUS_BUNDLE_NAMES)
        )
    target = ROOT / 'assets/checkpoints/beyondretarget_plus_8robots.pth'
    if target.is_file() and not force:
        if target.stat().st_size != PLUS_BUNDLE_SIZE or sha256(target) != PLUS_BUNDLE_SHA256:
            raise RuntimeError('Existing Plus bundle failed size/SHA validation; use --force')
        return
    if int(entry.get('size') or 0) not in (0, PLUS_BUNDLE_SIZE):
        raise RuntimeError('Drive Plus bundle has an unexpected size')
    download_file(str(entry['id']), target)
    if target.stat().st_size != PLUS_BUNDLE_SIZE or sha256(target) != PLUS_BUNDLE_SHA256:
        target.unlink(missing_ok=True)
        raise RuntimeError('Downloaded Plus bundle failed size/SHA validation')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-url", default=DEFAULT_FOLDER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    entries = folder_entries(args.folder_url)
    for name in ("yolov8x.pt", "robot.zip"):
        if name not in entries:
            raise RuntimeError(f"{name} is not visible in the public Base Drive folder")
    ensure_yolo(entries["yolov8x.pt"], args.force)
    ensure_robot(entries["robot.zip"], args.force)
    ensure_plus_bundle(entries, args.force)
    print("Shared Base/Plus assets are ready: assets/yolo/yolov8x.pt, assets/robot/, and assets/checkpoints/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
