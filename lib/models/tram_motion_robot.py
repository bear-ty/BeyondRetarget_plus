"""Decoder6 + final official motion memory, with immutable feature caches.

No ViT replay, no second disk feature cache, no GT conditions. Replay the
official pose readout + 16-frame motion Transformer in FP32/eval/no_grad.
Video inference taps the SAME per-window hidden states, never re-windows
already-stitched decoder tokens. All human/root outputs stay unchanged.
"""
import math
import numpy as np
import torch
from torch import nn
from pytorch3d.transforms import matrix_to_rotation_6d
from lib.models.modules import temporal_attention
from lib.models.hmr_vimo import HMR_VIMO
from lib.models.tram_robot import TramRobotAdapter
from lib.utils.geometry import rot6d_to_rotmat_hmr2

ARCHITECTURE = 'decoder6_motion_v6'
MOTION_CONTEXT_VERSION = 'tram_decoder6_motion384_fusion_v6'
MOTION_STATIC_VERSION = 'tram_decoder6_motion384_static6_v2'


def bbox_information(batch):
    center, scale = batch['center'], batch['scale']
    shape = center.shape[:-1]
    return HMR_VIMO.bbox_est(None, center.reshape(-1, 2), scale.reshape(-1),
        batch['img_focal'].reshape(-1), batch['img_center'].reshape(-1, 2)).reshape(*shape, 3)


class FrozenMotionReplay(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.decpose = nn.Linear(1024, 144)
        self.register_buffer('init_body_pose', torch.zeros(1, 144))
        self.motion = temporal_attention(in_dim=147, out_dim=144, hdim=384,
                                         nlayer=6, nhead=4, residual=False)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload.get('human', payload.get('state_dict', payload.get('model', payload)))
        if not isinstance(state, dict):
            raise ValueError('Human checkpoint does not contain a state dictionary')
        selected = {}
        for key, value in state.items():
            while key.startswith(('module.', 'model.', 'human.')):
                key = key.split('.', 1)[1]
            if key.startswith('smpl_head.decpose.'):
                selected[key.replace('smpl_head.', '', 1)] = value
            elif key == 'smpl_head.init_body_pose':
                selected['init_body_pose'] = value
            elif key.startswith('motion_module.'):
                selected[key.replace('motion_module.', 'motion.', 1)] = value
        self.load_state_dict(selected, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, h6, bbox_info):
        if h6.ndim != 3 or h6.shape[1:] != (16, 1024):
            raise ValueError('Replay needs a SAME-context [B,16,1024] decoder window')
        if bbox_info.shape != (*h6.shape[:2], 3):
            raise ValueError('Missing row-matched bbox conditioning')
        with torch.autocast(device_type=h6.device.type, enabled=False):
            pose0 = self.decpose(h6.float()) + self.init_body_pose
            x = torch.cat((pose0, bbox_info.float()), -1).permute(1, 0, 2)
            hidden = self.motion.trans(self.motion.pos_embedding(self.motion.l1(x)))
            pose = self.motion.l2(hidden)
        return hidden.permute(1, 0, 2).contiguous(), pose.permute(1, 0, 2).contiguous()


class DecoderMotionFusion(nn.Module):
    """Normalize projected branches before a learned .15/.85 scalar mixture."""
    def __init__(self, hidden=512, motion_weight=.85):
        super().__init__()
        if not 0 < motion_weight < 1:
            raise ValueError('motion_weight must be strictly between zero and one')
        self.decoder_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden),
                                           nn.GELU(), nn.LayerNorm(hidden))
        self.motion_proj = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden),
                                          nn.GELU(), nn.LayerNorm(hidden))
        self.motion_logit = nn.Parameter(torch.tensor(math.log(motion_weight/(1-motion_weight))))

    def weights(self):
        w = self.motion_logit.sigmoid()
        return {'fusion_motion_weight': w, 'fusion_decoder6_weight': 1-w}

    def forward(self, h6, motion):
        if h6.shape[:-1] != motion.shape[:-1] or h6.shape[-1] != 1024 or motion.shape[-1] != 384:
            raise ValueError('Decoder6/motion features must refer to identical frames')
        with torch.autocast(device_type=h6.device.type, enabled=False):
            w = self.motion_logit.sigmoid()
            return (1-w)*self.decoder_proj(h6.float()) + w*self.motion_proj(motion.float())


def stitch_motion_windows(windows, frame_ids):
    """Exactly mirror official first/center/last selection, including gaps."""
    frame_ids = np.asarray(frame_ids)
    if len(frame_ids) == 0 or np.any(np.diff(frame_ids) <= 0):
        raise ValueError('Expected strictly ordered output frame IDs')
    cuts = np.r_[0, np.flatnonzero(np.diff(frame_ids) != 1)+1, len(frame_ids)]
    parts, cursor = [], 0
    for left, right in zip(cuts[:-1], cuts[1:]):
        n = int(right-left)
        if n < 16:
            raise ValueError('Official inference should discard chunks shorter than 16')
        for start in range(n-15):
            if cursor >= len(windows):
                raise ValueError('Missing motion forward hook output')
            h = windows[cursor]
            if h.shape != (16, 384):
                raise ValueError('Video hook expected one 16-frame window')
            offsets = slice(None) if n == 16 else slice(0, 9) if start == 0 else slice(8, 16) if start == n-16 else slice(8, 9)
            parts.append(h[offsets]); cursor += 1
    if cursor != len(windows):
        raise ValueError('Unexpected extra motion windows')
    result = torch.cat(parts)
    if len(result) != len(frame_ids):
        raise AssertionError('Motion/frame association mismatch')
    return result


class TramMotionRobotAdapter(TramRobotAdapter):
    context_version = MOTION_CONTEXT_VERSION

    def __init__(self, cfg, robot_spec=None, with_human=True):
        super().__init__(cfg, robot_spec, with_human)
        if not cfg.MODEL.MOTION_MODULE or cfg.MODEL.MOTION_HDIM != 384 or cfg.MODEL.MOTION_NLAYER != 6:
            raise ValueError('Expected the released official six-layer motion module')
        self.layer_fusion = DecoderMotionFusion(int(cfg.ROBOT.HIDDEN_DIM),
                                                float(cfg.ROBOT.get('MOTION_INIT_WEIGHT', .85)))
        self.context_proj = nn.Identity()
        self.frozen_motion = None if with_human else FrozenMotionReplay(cfg.MODEL.CHECKPOINT)
        self.set_stage(cfg.ROBOT.STAGE)

    def set_stage(self, stage='head_only'):
        if stage!='head_only':
            raise ValueError('Motion-v6 Stage2 uses the separate shared six-joint head, not contact_only')
        super().set_stage(stage)
        if getattr(self, 'frozen_motion', None) is not None:
            self.frozen_motion.requires_grad_(False).eval()

    def _decode(self, human, valid=None):
        with torch.autocast(device_type=human['decoder_layer6'].device.type, enabled=False):
            visual = self.layer_fusion(human['decoder_layer6'], human['motion_hidden'])
            pose = human['smpl_pose6d'].float()
            b, t = pose.shape[:2]
            local = rot6d_to_rotmat_hmr2(pose.reshape(b, t, 24, 6)[:, :, 1:])
            condition = torch.cat((matrix_to_rotation_6d(local).reshape(b, t, 138),
                                   human['smpl_betas'].float()), -1)
            embedding = self.fusion(torch.cat((visual, self.pose_proj(condition)), -1))
            dof = self.dof_head(self.dof_norm(embedding+self.dof_adapter(embedding))) + self.neutral_dof
            if valid is not None:
                dof = torch.where(valid[..., None], dof, self.neutral_dof)
            flags = self.contact_head(embedding).float()
        return dict(dof=dof, root_pos=human['root_pos'], root_rot_6d=human['root_rot_6d'],
                    static_logits=flags[..., :2], contact_logits=flags[..., 2:])

    def forward(self, batch, return_human=True):
        if 'human_decoder_layer6' in batch:
            if self.frozen_motion is None:
                raise ValueError('Cached training requires FrozenMotionReplay')
            decoded = {k: batch['human_'+k] for k in ('decoder_layer6', 'smpl_pose6d',
                        'smpl_betas', 'root_pos', 'root_rot_6d')}
            hidden, pose = self.frozen_motion(decoded['decoder_layer6'], bbox_information(batch))
            decoded.update(motion_hidden=hidden, smpl_pose6d=pose)
            out = None
        else:
            captures = []
            hook = self.human.motion_module.l2.register_forward_pre_hook(
                lambda module, args: captures.append(args[0].detach().permute(1, 0, 2).contiguous()))
            try:
                out, decoded = self._human_forward(batch)
            finally:
                hook.remove()
            if len(captures) != 1:
                raise AssertionError('Expected one human forward')
            decoded['motion_hidden'] = captures[0]
        robot = self._decode(decoded, batch.get('frame_mask'))
        return dict(robot=robot, human=out, human_decoded=decoded) if return_human else robot

    @torch.inference_mode()
    def infer_sequence(self, imgfiles, boxes, img_focal=None, img_center=None,
                       valid=None, frame=None, device='cuda', static_predictor=None,
                       camera_world_to_cam_R=None):
        if self.human is None:
            raise ValueError('Video inference requires the official human model')
        self.eval(); windows = []; camera_prob_windows = []
        camera_static = static_predictor is not None and getattr(static_predictor,'requires_camera',False)
        if camera_static:
            from lib.util.beyondretarget.camera_condition import camera_relative_6d, continuous_window_frames, stitch_native_windows
            if camera_world_to_cam_R is None or np.shape(camera_world_to_cam_R)!=(len(imgfiles),3,3):
                raise ValueError('Camera-conditioned static head requires row-matched full-video camera rotations')
            native_frames=continuous_window_frames(np.arange(len(imgfiles)) if frame is None else frame,valid)
            # HMR_VIMO.inference_chunk calls self.forward directly: module forward
            # hooks are NOT fired. The existing SMPL head and motion submodule
            # hooks do fire, so capture decoder6, bbox and motion at those sites.
            camera_capture={}
            def capture_decoder(module,args,output):
                camera_capture['h6']=output[3]['decoder_layer6'].detach()
            def capture_motion_input(module,args):
                camera_capture['bbox_info']=args[0][...,-3:].detach()
            static_hooks=[self.human.smpl_head.register_forward_hook(capture_decoder),
                          self.human.motion_module.register_forward_pre_hook(capture_motion_input)]
        def capture(module, args):
            h = args[0]
            if h.shape != (16, 1, 384):
                raise ValueError('Unexpected official video-window shape')
            windows.append(h[:, 0].detach().cpu())
            if camera_static:
                index=len(camera_prob_windows)
                if index>=len(native_frames):raise AssertionError('Extra official inference window')
                ids=native_frames[index]
                cond=torch.from_numpy(camera_relative_6d(np.asarray(camera_world_to_cam_R)[ids])).to(device)[None]
                h6=camera_capture.pop('h6').reshape(1,16,1024)
                bbox=camera_capture.pop('bbox_info').reshape(1,16,3)
                logits=static_predictor(h6,motion_hidden=h.permute(1,0,2),bbox_info=bbox,camera_rot6d=cond)
                camera_prob_windows.append(logits[0].sigmoid().cpu())
        hook = self.human.motion_module.l2.register_forward_pre_hook(capture)
        try:
            human = self.human.inference(imgfiles, boxes, img_focal, img_center, valid, frame, device)
        finally:
            hook.remove()
            if camera_static:
                for handle in static_hooks:handle.remove()
        if human is None:
            return None
        n = len(human['frame'])
        hidden = stitch_motion_windows(windows, human['frame'].numpy())
        root = matrix_to_rotation_6d(human['pred_rotmat'][:, 0].float())
        frozen = dict(decoder_layer6=human['decoder_layer6'], motion_hidden=hidden,
            smpl_pose6d=human['pred_pose'], smpl_betas=human['pred_shape'],
            root_pos=human['pred_trans'].reshape(n, 3), root_rot_6d=root)
        dofs, probabilities = [], []
        for start in range(0, n, 256):
            chunk = {k: v[start:start+256].to(device)[None] for k, v in frozen.items()}
            dofs.append(self._decode(chunk)['dof'][0].cpu())
            if static_predictor is not None and not camera_static:
                if not getattr(static_predictor, 'uses_motion', False):
                    raise ValueError('New architecture requires the matching decoder6/motion static head')
                probabilities.append(static_predictor(chunk['decoder_layer6'][0],
                    motion_hidden=chunk['motion_hidden'][0]).sigmoid().cpu())
        result = dict(dof=torch.cat(dofs), root_pos=frozen['root_pos'], root_rot_6d=root,
            frame=human['frame'], pred_pose=human['pred_pose'], pred_shape=human['pred_shape'],
            pred_cam=human['pred_cam'], pred_trans=human['pred_trans'])
        if probabilities:
            result['static_prob6'] = torch.cat(probabilities)
        if camera_static:
            result['static_prob6']=stitch_native_windows(camera_prob_windows,human['frame'].numpy(),6)
        return result


class TramMotionStaticHead(nn.Module):
    uses_motion = True
    static_version = MOTION_STATIC_VERSION
    def __init__(self, checkpoint=None, motion_weight=.85):
        super().__init__()
        self.layer_fusion = DecoderMotionFusion(512, motion_weight)
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 6))
        self.frozen_motion = FrozenMotionReplay(checkpoint) if checkpoint else None

    def forward(self, h6, motion_hidden=None, bbox_info=None):
        if motion_hidden is None:
            if self.frozen_motion is None:
                raise ValueError('Provide motion features from the SAME official window')
            motion_hidden, _ = self.frozen_motion(h6, bbox_info)
        with torch.autocast(device_type=h6.device.type, enabled=False):
            return self.mlp(self.layer_fusion(h6.detach(), motion_hidden.detach()))


def build_robot_model(cfg, robot_spec=None, with_human=True):
    architecture = cfg.ROBOT.get('ARCHITECTURE', 'decoder46_v5')
    if architecture == ARCHITECTURE:
        return TramMotionRobotAdapter(cfg, robot_spec, with_human)
    if architecture != 'decoder46_v5':
        raise ValueError('Unknown architecture: '+architecture)
    return TramRobotAdapter(cfg, robot_spec, with_human)
