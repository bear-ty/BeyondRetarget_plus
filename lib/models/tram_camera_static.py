"""Independent camera-conditioned static predictor; immutable human replay.

Frozen decoder6/motion -> learned .15/.85 fusion + camera/bbox embeddings
-> two small 512-D temporal blocks -> 512/512/6 MLP. Native context = 16.
"""
import math
import torch
from torch import nn
from lib.models.tram_motion_robot import DecoderMotionFusion, FrozenMotionReplay

CAMERA_STATIC_VERSION = 'tram_decoder6_motion384_camera_static6_v3'


class TramCameraStaticHead(nn.Module):
    uses_motion = True
    requires_camera = True
    static_version = CAMERA_STATIC_VERSION

    def __init__(self, checkpoint=None, motion_weight=.85, layers=2, dropout=.1):
        super().__init__()
        if layers != 2:
            raise ValueError('This checkpoint contract uses exactly two temporal layers')
        self.layer_fusion = DecoderMotionFusion(512, motion_weight)
        self.camera_embed = nn.Sequential(nn.Linear(6, 512), nn.GELU(), nn.Linear(512, 512))
        self.bbox_embed = nn.Sequential(nn.Linear(3, 512), nn.GELU(), nn.Linear(512, 512))
        self.input_norm = nn.LayerNorm(512)
        block = nn.TransformerEncoderLayer(512, 8, dim_feedforward=1024,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(block, num_layers=2, norm=nn.LayerNorm(512),
                                               enable_nested_tensor=False)
        # Independent initialization: TransformerEncoder clones one template.
        for layer in self.temporal.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 6))
        position = torch.arange(16).float()[:, None]
        frequency = torch.exp(torch.arange(0, 512, 2).float()*(-math.log(10000.)/512))
        pe = torch.zeros(16, 512);pe[:, 0::2] = torch.sin(position*frequency);pe[:, 1::2] = torch.cos(position*frequency)
        self.register_buffer('position_encoding', pe[None])
        self.register_buffer('identity_rotation6d', torch.tensor([1.,0,0,0,1.,0]))
        self.frozen_motion = FrozenMotionReplay(checkpoint) if checkpoint else None

    def forward(self, h6, motion_hidden=None, bbox_info=None, camera_rot6d=None):
        if h6.ndim != 3 or h6.shape[1:] != (16, 1024):
            raise ValueError('Camera static head must see native [B,16,1024] windows')
        if bbox_info is None or bbox_info.shape != (*h6.shape[:2], 3):
            raise ValueError('Missing same-window bbox/intrinsic condition')
        if camera_rot6d is None or camera_rot6d.shape != (*h6.shape[:2], 6):
            raise ValueError('Missing same-window camera relative rotation')
        if motion_hidden is None:
            if self.frozen_motion is None:
                raise ValueError('Supply native-window frozen motion features')
            motion_hidden, _ = self.frozen_motion(h6.detach(), bbox_info.detach())
        # The human representation is always detached, including caller-supplied features.
        fusion = self.layer_fusion(h6.detach(), motion_hidden.detach())
        camera = (camera_rot6d.detach().float()-self.identity_rotation6d)/.1
        hidden = self.input_norm(fusion+self.camera_embed(camera)+self.bbox_embed(bbox_info.detach().float()))
        hidden = self.temporal(hidden+self.position_encoding)
        return self.mlp(hidden).float()
