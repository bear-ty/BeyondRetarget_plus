#!/usr/bin/env python3
"""Build the distributable ``beyondretarget_plus_8robots.pth`` checkpoint.

The bundle contains only the eight robot heads supported by the base release
and the camera-conditioned shared static head. It intentionally excludes
optimizers, training RNG state, experimental Stage-3/root checkpoints, and
robots without a public asset package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROBOTS = (
    "g1",
    "r1",
    "gr1t1_without_hand",
    "gr2v3_8_7_dummy_hand",
    "h1_with_hand_without_hand",
    "t1_serial",
    "tienkung",
    "atlas_v4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_robot_checkpoint(root: Path, robot: str) -> Path:
    matches = sorted(root.glob(f"stage1_{robot}_head_only/*/checkpoints/best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one best.pt for {robot}, found {len(matches)}")
    return matches[0]


def compact_robot(path: Path, robot: str, project_root: Path | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("robot_name") != robot or payload.get("stage") != "head_only":
        raise ValueError(f"Unexpected checkpoint contract in {path}")
    if payload.get("context_version") != "tram_decoder6_motion384_fusion_v6":
        raise ValueError(f"Unexpected decoder context in {path}")
    required = {"neutral_dof", "dof_head.weight", "contact_head.4.weight"}
    if not required.issubset(payload["model"]):
        raise ValueError(f"Incomplete robot head in {path}")
    provenance = dict(payload.get("provenance", {}))
    provenance.pop("human_checkpoint", None)
    provenance["human_checkpoint"] = "data/pretrain/vimo_checkpoint.pth.tar"
    if project_root is not None:
        spec_path = project_root / "configs" / f"robot_{robot}.yaml"
        if spec_path.is_file():
            provenance["robot_spec_sha256"] = sha256(spec_path)
            provenance["robot_spec_path"] = f"configs/robot_{robot}.yaml"
    return {
        "robot_name": robot,
        "stage": payload["stage"],
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "selection_score": payload.get("best"),
        "model": payload["model"],
        "provenance": provenance,
        "config": "decoder6_motion_v6",
        "source": path.name,
        "source_sha256": sha256(path),
    }


def compact_static(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("static_version") != "tram_decoder6_motion384_camera_static6_v3":
        raise ValueError("Only the audited camera-conditioned static-v3 checkpoint is supported")
    if payload.get("context_version") != "tram_decoder46_raw1024_smpl_hmr2col6d_v2":
        raise ValueError("Static checkpoint context does not match the Plus inference contract")
    if payload.get("optimizer_steps", 0) < 1:
        raise ValueError("Static checkpoint has not completed an optimizer step")
    def sanitize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "path" and isinstance(item, str):
                    result[key] = Path(item).name
                else:
                    result[key] = sanitize(item)
            return result
        if isinstance(value, str) and (value.startswith("/data1/") or value.startswith("/nasdata/") or value.startswith("/home/")):
            return Path(value).name
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(sanitize(item) for item in value)
        return value

    compact = sanitize({key: value for key, value in payload.items() if key not in {"optimizer", "source"}})
    compact["source"] = path.name
    return compact


def load_human_state(path: Path) -> tuple[dict, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("human", payload.get("state_dict", payload.get("model", payload)))
    if not isinstance(state, dict):
        raise ValueError(f"Human checkpoint has no state dictionary: {path}")
    required = ("smpl_head.decpose.weight", "backbone.patch_embed.proj.weight")
    keys = set(state)
    if not any(key in keys or key.removeprefix("module.") in keys for key in required):
        raise ValueError(f"Human checkpoint is not a compatible TRAM/VIMO model: {path}")
    return state, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--human-checkpoint", type=Path, required=True,
                        help="Complete official TRAM/VIMO checkpoint embedded in the Plus bundle")
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, help="Plus checkout used for release provenance hashes")
    args = parser.parse_args()

    human_state, human_sha256 = load_human_state(args.human_checkpoint)
    robots = {robot: compact_robot(one_robot_checkpoint(args.experiment_root, robot), robot, args.project_root)
              for robot in ROBOTS}
    for entry in robots.values():
        entry["provenance"]["human_checkpoint_sha256"] = human_sha256
    static = compact_static(args.static_checkpoint)
    bundle = {
        "bundle_format": "beyondretarget_plus_decoder6_bundle_v2",
        "bundle_version": 2,
        "architecture": "decoder6_motion_v6",
        "context_version": "tram_decoder6_motion384_fusion_v6",
        "cache_context_version": "tram_decoder46_raw1024_smpl_hmr2col6d_v2",
        "human": human_state,
        "human_provenance": {
            "sha256": human_sha256,
            "source": "official TRAM/VIMO checkpoint",
        },
        "robots": robots,
        "static_head": static,
        "native_root": {
            "included": False,
            "kind": "frozen_tram_native_root",
            "checkpoint_role": "human_vimo_checkpoint",
            "description": "Root pose remains in the embedded frozen official TRAM/VIMO human checkpoint; no learned Stage-3 root head is included.",
            "postprocess": "CRC-1 full reprojection",
        },
        "exclusions": {
            "robots": ["bluewhale", "penguin_15dof", "romeo_without_hand"],
            "stage3_root_checkpoints": "excluded",
            "old_static_v2": "excluded",
            "optimizers_and_schedulers": "excluded",
            "training_rng_states": "excluded",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    manifest = {
        "bundle": args.output.name,
        "sha256": digest,
        "bundle_format": bundle["bundle_format"],
        "contains_human_backbone": True,
        "human_checkpoint_sha256": human_sha256,
        "robots": list(ROBOTS),
        "static_version": static["static_version"],
        "native_root": bundle["native_root"],
        "exclusions": bundle["exclusions"],
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "sha256": digest, "robots": list(ROBOTS)}, indent=2))


if __name__ == "__main__":
    main()
