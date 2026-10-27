"""Independent, shared six-joint static head. Never changes a Stage 1 module."""
import torch
from torch import nn
from lib.models.tram_robot import DecoderLayerFusion
from lib.util.beyondretarget.feature_schema import CONTEXT_VERSION

STATIC_VERSION = 'tram_decoder46_world_static6_v1'
JOINT_IDS = (7, 10, 8, 11, 20, 21)
JOINT_NAMES = ('left_ankle', 'left_foot', 'right_ankle', 'right_foot', 'left_wrist', 'right_wrist')


class TramStaticHead(nn.Module):
    """Frozen temporal tokens -> independent .15/.85 fusion -> 1024/512/6 MLP.

    Like GVHMR's 512/512/6 GELU MLP, with an input projection adapted to
    TRAM's 1024-D tokens. No GT SMPL, robot DoF, or root is an input.
    """
    static_version = STATIC_VERSION
    def __init__(self):
        super().__init__()
        self.layer_fusion = DecoderLayerFusion(.85)
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 6))

    def forward(self, h4, h6):
        if h4.shape != h6.shape or h4.shape[-1] != 1024:
            raise ValueError('Expected matching [...,1024] frozen decoder tokens')
        return self.mlp(self.layer_fusion(h4.detach().float(), h6.detach().float()))


def _build_static_head(checkpoint, device='cpu', expected_human_sha256=None):
    from lib.models.tram_motion_robot import MOTION_STATIC_VERSION, TramMotionStaticHead
    from lib.models.tram_camera_static import CAMERA_STATIC_VERSION, TramCameraStaticHead
    if (checkpoint.get('static_version') not in (STATIC_VERSION,MOTION_STATIC_VERSION,CAMERA_STATIC_VERSION)
            or checkpoint.get('context_version') != CONTEXT_VERSION
            or checkpoint.get('optimizer_steps', 0) < 1
            or tuple(checkpoint.get('joint_ids', ())) != JOINT_IDS):
        raise ValueError('Not a trained, compatible six-joint static checkpoint')
    if expected_human_sha256 is not None and checkpoint.get('human_checkpoint_sha256') != expected_human_sha256:
        raise ValueError('Static head and frozen human checkpoint differ')
    if checkpoint['static_version']==CAMERA_STATIC_VERSION:
        bedlam_manifest = checkpoint.get('label_manifests',{}).get('bedlam_vid',{})
        position_policy = bedlam_manifest.get('position_policy')
        if position_policy is None:
            position_policy = bedlam_manifest.get('contract', {}).get('position_policy')
        if position_policy != 'native_smpl_true_world_from_gvhmr_v2':
            raise ValueError('Camera static head requires verified true-world BEDLAM labels and camera provenance')
        settings=checkpoint['settings']
        model=TramCameraStaticHead(layers=settings.get('static_temporal_layers',2),
                                  dropout=settings.get('static_dropout',.1))
    else:
        model=TramMotionStaticHead() if checkpoint['static_version']==MOTION_STATIC_VERSION else TramStaticHead()
    model=model.to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval(), checkpoint


def load_static_head(path, device='cpu', expected_human_sha256=None):
    from pathlib import Path
    if (Path(path).resolve().parent.parent/'INVALID_CAMERA_CONDITION.json').exists():
        raise ValueError('This interrupted audit run used auxiliary cam_ext, not true camera R. Do not use its checkpoint.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint.get('static_head'), dict):
        static_blob = checkpoint['static_head']
        if isinstance(static_blob.get('model'), dict):
            metadata = {key: value for key, value in static_blob.items() if key != 'model'}
            metadata['model'] = static_blob['model']
        else:
            metadata = dict(checkpoint.get('static_metadata') or {})
            metadata['model'] = static_blob
        checkpoint = metadata
    return _build_static_head(checkpoint, device, expected_human_sha256)


def load_static_head_from_payload(payload, device='cpu', expected_human_sha256=None):
    """Load the shared static head embedded in a Plus unified checkpoint."""
    if not isinstance(payload, dict) or not isinstance(payload.get('static_head'), dict):
        raise ValueError('Checkpoint does not contain an embedded Plus static head')
    static_blob = payload['static_head']
    if isinstance(static_blob.get('model'), dict):
        checkpoint = {key: value for key, value in static_blob.items() if key != 'model'}
        checkpoint['model'] = static_blob['model']
    else:
        checkpoint = dict(payload.get('static_metadata') or {})
        checkpoint['model'] = static_blob
    return _build_static_head(checkpoint, device, expected_human_sha256)


def build_static_head(settings, with_replay=True):
    architecture=settings.get('architecture','decoder46_v5')
    if architecture=='decoder46_v5':return TramStaticHead()
    if architecture!='decoder6_motion_v6':raise ValueError('Unknown static architecture: '+architecture)
    from lib.models.tram_motion_robot import TramMotionStaticHead
    from lib.core.config import update_cfg
    checkpoint=update_cfg(settings['base_config']).MODEL.CHECKPOINT if with_replay else None
    if settings.get('camera_conditioned',False):
        from lib.models.tram_camera_static import TramCameraStaticHead
        return TramCameraStaticHead(checkpoint,settings.get('motion_init_weight',.85),
            settings.get('static_temporal_layers',2),settings.get('static_dropout',.1))
    return TramMotionStaticHead(checkpoint,settings.get('motion_init_weight',.85))


def forward_static(model, item):
    raw=model.module if hasattr(model,'module') else model
    if getattr(raw,'requires_camera',False):
        return model(item['h6'],bbox_info=item['bbox_info'],camera_rot6d=item['camera_rot6d'])
    if getattr(raw,'uses_motion',False):return model(item['h6'],bbox_info=item['bbox_info'])
    return model(item['h4'],item['h6'])


def static_fusion_weights(model):
    if hasattr(model.layer_fusion,'weights'):
        return {k:float(v.detach()) for k,v in model.layer_fusion.weights().items()}
    return {'layer6_weight':float(model.layer_fusion.layer6_logit.detach().sigmoid())}
