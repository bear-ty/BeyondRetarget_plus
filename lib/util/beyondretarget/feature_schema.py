"""Raw frozen decoder tokens. No trainable fusion is baked into this cache."""
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

CONTEXT_VERSION = 'tram_decoder46_raw1024_smpl_hmr2col6d_v2'
CACHE_FIELDS = {
    'decoder_layer4': 1024,
    'decoder_layer6': 1024,
    'smpl_pose6d': 144,
    'smpl_betas': 10,
    'pred_cam': 3,
    'root_pos': 3,
    'root_rot_6d': 6,
}


def pack_frozen_outputs(out, batch_size, length):
    b, t = batch_size, length
    return {
        'decoder_layer4': out['decoder_layer4'].reshape(b, t, 1024).float(),
        'decoder_layer6': out['decoder_layer6'].reshape(b, t, 1024).float(),
        # Official post-motion SMPL pose: HMR2 column-6D, NOT row-6D.
        'smpl_pose6d': out['pred_pose'].reshape(b, t, 144).float(),
        'smpl_betas': out['pred_shape'].reshape(b, t, 10).float(),
        'pred_cam': out['pred_cam'].reshape(b, t, 3).float(),
        'root_pos': out['trans_full'].reshape(b, t, 3).float(),
        'root_rot_6d': matrix_to_rotation_6d(out['pred_rotmat'][:, 0].float()).reshape(b, t, 6),
    }


def validate_shard(payload, first_row, length=16):
    if payload.get('first_row') != first_row or payload.get('length') != length:
        raise ValueError('Frozen-feature frame association mismatch')
    for key, width in CACHE_FIELDS.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (length, width):
            raise ValueError(f'Invalid frozen {key}: expected {(length, width)}')
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Non-finite/non-float32 frozen {key}')
