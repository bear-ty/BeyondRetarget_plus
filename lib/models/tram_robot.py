"""Frozen TRAM decoder layers 4/6 + GVRMR-style lightweight robot head.

TRAM human rotations remain HMR2-column 6D internally. Robot outputs ALWAYS
derive row-6D from the decoded rotation MATRIX; no reinterpretation of 6D.
Roots are camera-frame SMPL roots; XML local basis is applied only at FK/export.
"""
import math
import torch
from torch import nn
from pytorch3d.transforms import matrix_to_rotation_6d
from .hmr_vimo import HMR_VIMO
from lib.robot.robot_spec import load_robot_spec
from lib.utils.geometry import rot6d_to_rotmat_hmr2
from lib.util.beyondretarget.feature_schema import CONTEXT_VERSION, CACHE_FIELDS, pack_frozen_outputs


class DecoderLayerFusion(nn.Module):
    """Trainable scalar mixture; raw frozen features remain reusable."""
    def __init__(self, layer6_weight=0.85):
        super().__init__()
        w6 = float(layer6_weight)
        if not 0 < w6 < 1:
            raise ValueError('Initial layer6 weight must be strictly between 0 and 1')
        self.norm4 = nn.LayerNorm(1024)
        self.norm6 = nn.LayerNorm(1024)
        self.layer6_logit = nn.Parameter(torch.tensor(math.log(w6 / (1 - w6))))

    def forward(self, h4, h6):
        w6 = self.layer6_logit.sigmoid()
        return (1 - w6) * self.norm4(h4) + w6 * self.norm6(h6)


class TramRobotAdapter(nn.Module):
    def __init__(self,cfg,robot_spec=None,with_human=True):
        super().__init__()
        if not cfg.ROBOT.FREEZE_HUMAN:
            raise ValueError('Plus training freezes all official TRAM human parameters')
        if int(cfg.DATASET.SEQ_LEN)!=16:
            raise ValueError('Official VIMO checkpoint uses 16-frame temporal blocks')
        self.cfg = cfg
        self.human = HMR_VIMO(cfg=cfg) if with_human else None
        self.robot_spec = robot_spec or load_robot_spec(cfg.ROBOT.SPEC)
        self.seq_len = 16
        self.context_dim = 1024
        hidden = int(cfg.ROBOT.HIDDEN_DIM)
        self.layer_fusion = DecoderLayerFusion(cfg.ROBOT.get('LAYER6_INIT_WEIGHT', 0.85))
        self.context_proj = nn.Sequential(nn.LayerNorm(self.context_dim),nn.Linear(self.context_dim,hidden),nn.GELU())
        self.pose_proj = nn.Sequential(nn.LayerNorm(148), nn.Linear(148,hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.LayerNorm(2*hidden), nn.Linear(2*hidden,hidden),
                                    nn.GELU(), nn.Linear(hidden,hidden), nn.GELU())
        self.dof_adapter = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,int(cfg.ROBOT.ADAPTER_DIM)),
                                         nn.GELU(), nn.Linear(int(cfg.ROBOT.ADAPTER_DIM),hidden))
        self.dof_norm = nn.LayerNorm(hidden)
        self.dof_head = nn.Linear(hidden,self.robot_spec.dof)
        nn.init.xavier_uniform_(self.dof_head.weight,gain=float(cfg.ROBOT.OUTPUT_GAIN))
        nn.init.zeros_(self.dof_head.bias)
        neutral = torch.tensor(self.robot_spec.neutral_dof or [0.]*self.robot_spec.dof)
        if self.robot_spec.neutral_dof is not None and self.robot_spec.model_to_xml_dof is not None:
            neutral = neutral[torch.tensor(self.robot_spec.model_to_xml_dof)]
        if neutral.numel()!=self.robot_spec.dof:
            raise ValueError('Neutral pose/spec DoF mismatch')
        self.register_buffer('neutral_dof',neutral)
        self.contact_head = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,int(cfg.ROBOT.ADAPTER_DIM)),
                                         nn.GELU(),nn.Dropout(float(cfg.ROBOT.DROPOUT)),
                                         nn.Linear(int(cfg.ROBOT.ADAPTER_DIM),4))
        self.set_stage(cfg.ROBOT.STAGE)

    def _pose_modules(self):
        return (self.layer_fusion,self.context_proj,self.pose_proj,self.fusion,
                self.dof_adapter,self.dof_norm,self.dof_head)

    def set_stage(self,stage='head_only'):
        if stage not in ('head_only','contact_only'):
            raise ValueError('Only audited frozen-core head_only/contact_only stages are enabled')
        self.stage = stage
        if self.human is not None:
            self.human.requires_grad_(False)
            self.human.eval()
        for module in self._pose_modules():
            module.requires_grad_(stage=='head_only')
        self.contact_head.requires_grad_(stage=='contact_only')

    def train(self,mode=True):
        super().train(mode)
        if self.human is not None: self.human.eval()
        if self.stage=='contact_only':
            for module in self._pose_modules(): module.eval()
        return self

    @torch.no_grad()
    def _human_forward(self,batch):
        if self.human is None:
            raise ValueError('This adapter needs a verified frozen-feature cache')
        b,t = batch['img'].shape[:2]
        keys = ('img','center','scale','img_focal','img_center')
        flat = {k:batch[k].flatten(0,1) for k in keys}
        out,_ = self.human(flat)
        return out,pack_frozen_outputs(out,b,t)

    def _decode(self,human,valid=None):
        # As in GVRMR, keep the small bridge/FK path FP32 under mixed precision.
        with torch.autocast(device_type=human['decoder_layer6'].device.type,enabled=False):
            visual = self.layer_fusion(human['decoder_layer4'].float(),human['decoder_layer6'].float())
            pose = human['smpl_pose6d'].float()
            b,t = pose.shape[:2]
            # Exclude root rotation from DoF conditions. Convert by actual
            # rotation matrices, never reinterpret HMR2 column-6D as row-6D.
            local_matrix = rot6d_to_rotmat_hmr2(pose.reshape(b,t,24,6)[:,:,1:])
            local_pose = matrix_to_rotation_6d(local_matrix).reshape(b,t,138)
            condition = torch.cat((local_pose,human['smpl_betas'].float()),dim=-1)
            embedding = self.fusion(torch.cat((self.context_proj(visual),self.pose_proj(condition)),dim=-1))
            adapted = embedding + self.dof_adapter(embedding)
            dof = self.dof_head(self.dof_norm(adapted)) + self.neutral_dof
            if valid is not None:
                dof = torch.where(valid[...,None],dof,self.neutral_dof)
            flags = self.contact_head(embedding).float()
        return {'dof':dof,'root_pos':human['root_pos'],'root_rot_6d':human['root_rot_6d'],
                'static_logits':flags[...,:2],'contact_logits':flags[...,2:]}

    def forward(self,batch,return_human=True):
        if 'human_context' in batch:
            raise ValueError('Old pooled-feature cache is incompatible with decoder46 v2')
        if 'human_decoder_layer4' in batch:
            decoded = {key:batch['human_'+key] for key in CACHE_FIELDS}
            out = None
        else:
            out,decoded = self._human_forward(batch)
        robot = self._decode(decoded,batch.get('frame_mask'))
        return {'robot':robot,'human':out,'human_decoded':decoded} if return_human else robot

    @torch.inference_mode()
    def infer_sequence(self,imgfiles,boxes,img_focal=None,img_center=None,valid=None,frame=None,device='cuda',static_predictor=None):
        if self.human is None: raise ValueError('Video inference requires the official human backbone')
        self.eval()
        human = self.human.inference(imgfiles,boxes,img_focal,img_center,valid,frame,device)
        if human is None: return None
        n = len(human['frame'])
        pos = human['pred_trans'].reshape(n,3).to(device)
        root = matrix_to_rotation_6d(human['pred_rotmat'][:,0].to(device).float())
        frozen = {'decoder_layer4':human['decoder_layer4'], 'decoder_layer6':human['decoder_layer6'],
                  'smpl_pose6d':human['pred_pose'], 'smpl_betas':human['pred_shape'],
                  'pred_cam':human['pred_cam'], 'root_pos':pos, 'root_rot_6d':root}
        # The new robot head is framewise; temporal context was already encoded
        # by the frozen official model. No extra overlap/padding is necessary.
        result = {k:torch.zeros(n,w,device=device) for k,w in (('dof',self.robot_spec.dof),('static_logits',2),('contact_logits',2))}
        ids = human['frame']
        for start in range(0,n,256):
            stop = min(start+256,n)
            decoded = self._decode({key:value[start:stop].to(device)[None] for key,value in frozen.items()})
            for key in result: result[key][start:stop] = decoded[key][0]
        result = {k:v.cpu() for k,v in result.items()}
        result.update(root_pos=pos.cpu(),root_rot_6d=root.cpu(),frame=ids,
                      pred_pose=human['pred_pose'],pred_shape=human['pred_shape'],pred_cam=human['pred_cam'],
                      pred_trans=human['pred_trans'])
        # Independent Stage 2; no new Stage 1 parameters or checkpoint keys.
        if static_predictor is not None:
            probabilities=[]
            for start in range(0,n,256):
                logits=static_predictor(human['decoder_layer4'][start:start+256].to(device),
                                        human['decoder_layer6'][start:start+256].to(device))
                probabilities.append(logits.sigmoid().cpu())
            result['static_prob6']=torch.cat(probabilities)
        return result


def load_human_weights(model,path=None,payload=None):
    """Load frozen TRAM/VIMO weights from a standalone or unified checkpoint.

    A Plus release bundle stores the complete human backbone under ``human``;
    standalone TRAM/VIMO checkpoints keep it under ``model`` or ``state_dict``.
    """
    if payload is None:
        if path is None:
            raise ValueError('Provide either a checkpoint path or an in-memory payload')
        payload = torch.load(path,map_location='cpu',weights_only=False)
    state = payload.get('human', payload.get('state_dict',payload.get('model',payload)))
    if not isinstance(state, dict):
        raise ValueError('Human checkpoint does not contain a state dictionary')
    clean = {}
    for key,value in state.items():
        while key.startswith(('module.','model.','human.')):
            key = key.split('.',1)[1]
        clean[key] = value
    missing,unexpected = model.human.load_state_dict(clean,strict=False)
    parameters = set(dict(model.human.named_parameters()))
    bad = [k for k in missing if k in parameters]
    if bad: raise RuntimeError('Official checkpoint missing human parameters: '+str(bad[:20]))
    return missing,unexpected


def load_tram_robot_checkpoint(model,path,map_location='cpu',strict=True):
    payload = torch.load(path,map_location=map_location,weights_only=False)
    selected = payload
    robot_entries = payload.get('robots', payload.get('robot_heads'))
    if isinstance(robot_entries, dict):
        selected = robot_entries.get(model.robot_spec.name)
        if not isinstance(selected, dict):
            supported = ', '.join(sorted(robot_entries))
            raise ValueError(f'Unified checkpoint has no head for {model.robot_spec.name}; available: {supported}')
        selected = dict(selected)
        selected.setdefault('context_version', selected.get('provenance', {}).get('context_version',
                                                                                payload.get('context_version')))
        selected.setdefault('robot_name', model.robot_spec.name)
        selected.setdefault('bundle_format_version', payload.get('format_version'))
        selected.setdefault('bundle_architecture', payload.get('architecture'))
        selected.setdefault('bundle_context_version', payload.get('context_version'))
        if 'static_head' in payload:
            static_blob = payload['static_head']
            if isinstance(static_blob, dict) and isinstance(static_blob.get('model'), dict):
                selected['static_head'] = static_blob['model']
                selected['static_metadata'] = {k: v for k, v in static_blob.items() if k != 'model'}
            else:
                selected['static_head'] = static_blob
                selected['static_metadata'] = payload.get('static_metadata', {})
        if 'human' in payload:
            selected['human'] = payload['human']
            selected['human_provenance'] = payload.get('human_provenance', {})
    if selected.get('context_version')!=getattr(model,'context_version',CONTEXT_VERSION):
        raise ValueError('Checkpoint architecture/context differs from the requested model')
    if selected.get('robot_name')!=model.robot_spec.name:
        raise ValueError('Robot checkpoint/spec mismatch')
    missing,unexpected = model.load_state_dict(selected['model'],strict=False)
    missing = [k for k in missing if not k.startswith(('human.','frozen_motion.'))]
    if missing or unexpected:
        raise RuntimeError(f'Adapter checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    return selected,missing,unexpected
