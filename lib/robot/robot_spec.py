import os
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None
import yaml

from lib.robot.robot_kinematics import G1_NMR_TO_XML_JOINT_MAPPING


@dataclass(frozen=True)
class SharedJointSpec:
    name: str
    robot_link: str
    smpl_joint: int
    local_offset: Optional[List[float]] = None


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: str
    dof: int
    xml_dof: Optional[int]
    root_body: str
    model_dof_order: List[str]
    model_to_xml_dof: Optional[List[int]]
    shared_joint_map: List[SharedJointSpec]
    tail_shared_indices: List[int]
    shared_joint_parents: List[int]
    gamp_edge_scale: List[float]
    neutral_dof: Optional[List[float]]
    torso_spine2_proxy_alpha: Optional[float]
    foot_grounding_meshes: Dict[str, Dict[str, str]]


def load_robot_spec(path: str, project_root: Optional[str] = None) -> RobotSpec:
    # Robot specs are plain YAML; avoid making model construction depend on
    # OmegaConf's optional structured-config stack.
    with open(path, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    robot_root = os.environ.get('BRPLUS_ROBOT_ROOT', os.path.join(project_root, 'assets', 'robot'))
    xml_path = str(cfg["xml_path"])
    if not os.path.isabs(xml_path):
        marker = os.path.join('assets', 'robot') + os.sep
        if xml_path.startswith(marker):
            xml_path = os.path.join(robot_root, xml_path[len(marker):])
        else:
            xml_path = os.path.join(project_root, xml_path)
    shared = [
        SharedJointSpec(
            name=str(item["name"]),
            robot_link=str(item["robot_link"]),
            smpl_joint=int(item["smpl_joint"]),
            local_offset=None if item.get("local_offset") is None else [float(value) for value in item["local_offset"]],
        )
        for item in cfg["shared_joint_map"]
    ]
    model_to_xml = cfg.get("model_to_xml_dof")
    if isinstance(model_to_xml, str):
        if model_to_xml != "g1_nmr_to_xml":
            raise ValueError(f"Unsupported model_to_xml_dof mapping: {model_to_xml}")
        model_to_xml = G1_NMR_TO_XML_JOINT_MAPPING
    foot_grounding_meshes = {}
    for name, item in dict(cfg.get("foot_grounding_meshes", {})).items():
        mesh_path = str(item["mesh_path"])
        if not os.path.isabs(mesh_path):
            marker = os.path.join('assets', 'robot') + os.sep
            if mesh_path.startswith(marker):
                mesh_path = os.path.join(robot_root, mesh_path[len(marker):])
            else:
                mesh_path = os.path.abspath(os.path.join(project_root, mesh_path))
        foot_grounding_meshes[str(name)] = {"robot_link": str(item["robot_link"]), "mesh_path": mesh_path}
    return RobotSpec(
        name=str(cfg["name"]),
        xml_path=xml_path,
        dof=int(cfg["dof"]),
        xml_dof=None if cfg.get("xml_dof") is None else int(cfg["xml_dof"]),
        root_body=str(cfg.get("root_body", "")),
        model_dof_order=[str(name) for name in cfg.get("model_dof_order", [])],
        model_to_xml_dof=None if model_to_xml is None else [int(idx) for idx in model_to_xml],
        shared_joint_map=shared,
        tail_shared_indices=[int(idx) for idx in cfg.get("tail_shared_indices", [])],
        shared_joint_parents=[int(idx) for idx in cfg.get("shared_joint_parents", [])],
        gamp_edge_scale=[float(value) for value in cfg.get("gamp_edge_scale", [])],
        neutral_dof=None if cfg.get("neutral_dof") is None else [float(v) for v in cfg["neutral_dof"]],
        torso_spine2_proxy_alpha=None if cfg.get("torso_spine2_proxy_alpha") is None else float(cfg["torso_spine2_proxy_alpha"]),
        foot_grounding_meshes=foot_grounding_meshes,
    )


def model_dof_to_xml_dof(dof, robot_spec: RobotSpec):
    if robot_spec.model_to_xml_dof is None:
        return dof
    if dof.shape[-1] != len(robot_spec.model_to_xml_dof):
        raise ValueError(f"Expected {len(robot_spec.model_to_xml_dof)} dofs for {robot_spec.name}, got {dof.shape[-1]}")
    xml_dof_count = int(robot_spec.xml_dof or len(robot_spec.model_to_xml_dof))
    if robot_spec.neutral_dof is not None:
        neutral = dof.new_tensor(robot_spec.neutral_dof)
        if neutral.numel() != xml_dof_count:
            raise ValueError(f"Expected neutral_dof with {xml_dof_count} values for {robot_spec.name}, got {neutral.numel()}")
        xml_dof = neutral.view(*([1] * (dof.dim() - 1)), xml_dof_count).expand(*dof.shape[:-1], xml_dof_count).clone()
    else:
        xml_dof = dof.new_zeros(*dof.shape[:-1], xml_dof_count)
    xml_dof[..., robot_spec.model_to_xml_dof] = dof
    return xml_dof
