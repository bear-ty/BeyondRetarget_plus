"""Thin TRAM interface over the actual GVRMR v5 objective (see provenance)."""
import torch
import torch.nn.functional as F
from torch import nn
from lib.robot.robot_spec import load_robot_spec
from .gvrmr_v5_loss import GvrmrV5Loss, masked_mean
from .supervision import SmplRobotReference


class TramGvrmrV5Loss(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.stage = cfg.ROBOT.STAGE
        self.reference = SmplRobotReference()
        self.robot_loss = GvrmrV5Loss(load_robot_spec(cfg.ROBOT.SPEC), {}, device=device,
                                    target_scale_mode='gamp', use_torso_proxy=True,
                                    robot_to_target_coord='zup_to_yup',
                                    v5_config=dict(cfg.ROBOT_LOSS))

    def set_epoch(self, epoch):
        self.robot_loss.set_epoch(epoch)

    def forward(self, pred, batch, metrics=False):
        target = self.reference(batch)
        stats = (self.robot_loss.compute_metrics(pred,target) if metrics
                 else self.robot_loss(pred,target))
        stats.pop('v5_frozen_root_position_mm',None)
        if self.stage == 'contact_only':
            required = ('static_label','contact_label','static_mask','contact_mask')
            if any(k not in batch for k in required):
                raise ValueError('contact_only needs verified world static/contact labels and masks')
            contact_stats = {}
            for kind in ('static','contact'):
                mask = batch[kind+'_mask'].bool() & batch['frame_mask'][...,None].bool()
                if not mask.any():
                    raise ValueError('contact_only batch has no valid '+kind+' labels')
                logits, labels = pred[kind+'_logits'].float(), batch[kind+'_label'].float()
                value = masked_mean(F.binary_cross_entropy_with_logits(logits,labels,reduction='none'),mask)
                contact_stats['loss_'+kind+'_bce'] = value
                contact_stats[kind+'_accuracy'] = masked_mean(((logits>=0)==(labels>=.5)).float(),mask)
            stats.update(contact_stats)
            stats['loss'] = contact_stats['loss_static_bce']+contact_stats['loss_contact_bce']
            stats['contact_selection_score'] = stats['loss']
        return stats['loss'],stats
