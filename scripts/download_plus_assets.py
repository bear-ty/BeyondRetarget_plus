#!/usr/bin/env python3
"""Download a publisher-provided Plus asset manifest with SHA256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(value: str) -> dict:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(value, timeout=60) as response:
            payload = response.read()
    else:
        payload = Path(value).expanduser().read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("assets"), list):
        raise ValueError("manifest must be an object containing an assets list")
    return manifest


def validate_entry(entry: dict) -> tuple[str, str, str, int | None, str]:
    if not isinstance(entry, dict):
        raise ValueError("each manifest asset must be an object")
    name = str(entry.get("name", ""))
    url = str(entry.get("url", ""))
    path = str(entry.get("path", ""))
    digest = str(entry.get("sha256", "")).lower()
    if not name or not url or not path or len(digest) != 64:
        raise ValueError(f"invalid manifest entry: {entry!r}")
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"asset path must stay inside the project: {path}")
    size = entry.get("size")
    if size is not None:
        size = int(size)
        if size < 0:
            raise ValueError(f"invalid asset size for {name}")
    return name, url, path, size, digest


def download(entry: tuple[str, str, str, int | None, str], force: bool) -> None:
    name, url, relative, expected_size, expected_hash = entry
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError(f"refusing to overwrite symbolic link: {target}")
    if target.is_file() and not force:
        actual = sha256(target)
        if actual == expected_hash and (expected_size is None or target.stat().st_size == expected_size):
            print(f"[asset] verified {name}: {target}")
            return
        raise RuntimeError(f"existing asset failed validation: {target}; use --force to replace it")

    partial = target.with_name(target.name + ".partial")
    print(f"[asset] downloading {name} -> {target}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BeyondRetarget-Plus asset downloader"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"{name}: size mismatch ({partial.stat().st_size} != {expected_size})")
    actual = sha256(partial)
    if actual != expected_hash:
        raise RuntimeError(f"{name}: SHA256 mismatch ({actual} != {expected_hash})")
    os.replace(partial, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Local JSON file or HTTPS URL")
    parser.add_argument("--force", action="store_true", help="Replace existing files after validation")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    for raw in manifest["assets"]:
        download(validate_entry(raw), args.force)
    print("Plus assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
