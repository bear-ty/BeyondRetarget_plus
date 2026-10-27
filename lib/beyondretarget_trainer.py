"""V5-style experiment trainer: frozen core, DDP, resumable adapter checkpoints."""
import contextlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from lib.models.tram_robot import CONTEXT_VERSION,load_tram_robot_checkpoint
from lib.util.beyondretarget.gvrmr_v5_loss import GvrmrV5Loss

LOG = logging.getLogger(__name__)


def move_batch(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


class MetricMean:
    def __init__(self): self.total={}; self.count=0
    def add(self,stats,n):
        for k,v in stats.items():
            value=float(v.detach()) if torch.is_tensor(v) else float(v)
            if not math.isfinite(value): raise FloatingPointError('Non-finite metric '+k)
            self.total[k]=self.total.get(k,0.)+value*n
        self.count+=n
    def result(self,device):
        keys=sorted(self.total)
        if dist.is_initialized():
            allkeys=[None]*dist.get_world_size()
            dist.all_gather_object(allkeys,keys)
            keys=sorted(set(k for group in allkeys for k in group))
        vector=torch.tensor([self.total.get(k,0.) for k in keys]+[self.count],dtype=torch.float64,device=device)
        if dist.is_initialized(): dist.all_reduce(vector)
        return {k:float(vector[i]/vector[-1].clamp_min(1)) for i,k in enumerate(keys)}


def log_metrics(label,stats):
    if not stats: return
    value=lambda k:stats.get(k,float('nan'))
    LOG.info('METRICS %s: selection=%.5f (no PA)',label,value('v5_selection_score'))
    if 'v5_incam_root_relative_mm' in stats:
        LOG.info('  pose  local=%.2fmm | root-relative incam=%.2fmm | PA diagnostic=%.2fmm',
                 value('v5_local_mpjpe_mm'),value('v5_incam_root_relative_mm'),value('smpl_gamp_pa_mpjpe_mm'))
    else:
        LOG.info('  pose  local=%.2fmm | incam/PA diagnostics: validation only',value('v5_local_mpjpe_mm'))
    LOG.info('  axes  rotation=%.2fdeg | torso=%.2f | thigh=%.2f | shank=%.2f | arm=%.2f',
             value('v5_rotation_deg'),value('v5_torso_deg'),value('v5_thigh_deg'),
             value('v5_shank_deg'),value('v5_arm_deg'))
    LOG.info('  dynamics  velocity=%.4fm/s | accel=%.3fm/s^2 | DoF accel RMS=%.2frad/s^2',
             value('v5_velocity_error_m_s'),value('v5_acceleration_error_m_s2'),value('v5_dof_acc_rms_rad_s2'))
    LOG.info('  limits  outside=%.4f%% | frame observability GT/pred=%.3f/%.3f',
             value('v5_limit_over_rate')*100,value('v5_frame_valid_rate'),value('v5_pred_frame_valid_rate'))
    if 'v5_frozen_root_rotation_deg' in stats:
        LOG.info('  frozen root rotation error=%.2fdeg (diagnostic, not optimized)',value('v5_frozen_root_rotation_deg'))
    weighted=sum(stats.get('loss_v5_'+k+'_weighted',0.) for k in GvrmrV5Loss.term_names)
    LOG.info('  LOSS total=%.5f | temporal ramp=%.2f | full-strength=%.5f',
             value('loss'),value('v5_temporal_ramp'),value('v5_full_strength_loss'))
    for k in GvrmrV5Loss.term_names:
        LOG.info('    %-13s raw=%9.5f weighted=%9.5f share=%6.2f%%',k,
                 value('loss_v5_'+k),value('loss_v5_'+k+'_weighted'),
                 stats.get('loss_v5_'+k+'_weighted',0.)/max(weighted,1e-12)*100)
    if 'contact_selection_score' in stats:
        LOG.info('  static/contact BCE=%.5f/%.5f | accuracy=%.3f/%.3f',
                 value('loss_static_bce'),value('loss_contact_bce'),
                 value('static_accuracy'),value('contact_accuracy'))


class RobotTrainer:
    def __init__(self,cfg,model,loss_fn,loaders,optimizer,provenance):
        self.cfg,self.model,self.loss_fn=cfg,model,loss_fn
        self.raw=model.module if hasattr(model,'module') else model
        self.train_loader,self.val_loaders=loaders
        self.optimizer=optimizer
        self.provenance=provenance
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.device=next(self.raw.parameters()).device
        self.logdir=Path(cfg.LOGDIR)
        self.global_step,self.start_epoch,self.best=0,0,float('inf')
        effective=int(cfg.TRAIN.EFFECTIVE_BATCH_SIZE)
        micro=int(cfg.TRAIN.BATCH_SIZE)*self.world
        if effective%micro: raise ValueError('EFFECTIVE_BATCH_SIZE must be divisible by batch_size * world_size')
        self.accumulate=effective//micro
        if self.accumulate<1: raise ValueError('Effective batch cannot be smaller than micro batch')
        # GVRMR v5 configure_optimizers uses constant-LR AdamW, not cosine decay.
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        self.writer=None
        if not self.rank:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer=SummaryWriter(str(self.logdir/'tensorboard'))
            except ImportError:
                LOG.warning('TensorBoard unavailable; complete metrics are in metrics_history.jsonl')
        if cfg.TRAIN.RESUME: self._resume(cfg.TRAIN.RESUME)
        LOG.info('Batch: micro=%d per rank | world=%d | accumulation=%d | global effective=%d',
                 cfg.TRAIN.BATCH_SIZE,self.world,self.accumulate,effective)

    def _resume(self,path):
        saved,_,_=load_tram_robot_checkpoint(self.raw,path)
        if saved['stage']!=self.cfg.ROBOT.STAGE: raise ValueError('Use FROM_CHECKPOINT when changing stage')
        for key in ('human_checkpoint_sha256','context_version','robot_spec_sha256','robot_asset_sha256'):
            if saved['provenance'].get(key)!=self.provenance.get(key):
                raise ValueError('Resume provenance mismatch: '+key)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.start_epoch=saved['epoch']+1
        self.global_step=saved['global_step']
        self.best=saved['best']
        rng=saved.get('rng_by_rank',[])
        if len(rng)==self.world:
            state=rng[self.rank]
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
            if self.device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],self.device)
        LOG.info('Resumed epoch=%d step=%d best=%.5f',self.start_epoch,self.global_step,self.best)

    def _save(self,epoch,is_best,record):
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None}
        rngs=[rng]
        if dist.is_initialized():
            rngs=[None]*self.world;dist.all_gather_object(rngs,rng)
        if self.rank: return
        state={k:v.detach().cpu() for k,v in self.raw.state_dict().items() if not k.startswith(('human.','frozen_motion.'))}
        payload={'format_version':2,'context_version':getattr(self.raw,'context_version',CONTEXT_VERSION),'model':state,
                 'robot_name':self.raw.robot_spec.name,'stage':self.cfg.ROBOT.STAGE,
                 'epoch':epoch,'global_step':self.global_step,'best':self.best,
                 'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
                 'rng_by_rank':rngs,'metrics':record,'provenance':self.provenance,'config':self.cfg.dump()}
        folder=self.logdir/'checkpoints';folder.mkdir(exist_ok=True)
        for name in (['last.pt','best.pt'] if is_best else ['last.pt']):
            tmp=folder/(name+'.tmp')
            torch.save(payload,tmp);os.replace(tmp,folder/name)

    def _run(self,loader,training):
        self.model.train(training);self.loss_fn.train(training)
        mean=MetricMean()
        maximum=int(self.cfg.TRAIN.MAX_TRAIN_BATCHES if training else self.cfg.TRAIN.MAX_VAL_BATCHES)
        batches=min(len(loader),maximum) if maximum else len(loader)
        # Bounded iteration avoids reading one extra expensive image batch at the limit.
        import itertools
        for index,batch in enumerate(itertools.islice(loader,batches)):
            batch=move_batch(batch,self.device)
            if training and index%self.accumulate==0:
                self.optimizer.zero_grad(set_to_none=True)
                window_start=index
                window_end=min(index+self.accumulate,batches)
                total_examples=sum(min(loader.batch_size,len(loader.sampler)-j*loader.batch_size)
                                   for j in range(window_start,window_end))
            sync=not training or index+1==window_end
            ddp_context=(self.model.no_sync() if training and not sync and hasattr(self.model,'no_sync')
                         else contextlib.nullcontext())
            autocast=torch.autocast(self.device.type,dtype=torch.bfloat16,
                                   enabled=self.device.type=='cuda' and self.cfg.TRAIN.PRECISION=='bf16')
            with ddp_context,torch.set_grad_enabled(training):
                with autocast:
                    # Uneven per-source validation shards must not call DDP forward collectives.
                    pred=(self.model if training else self.raw)(batch,return_human=False)
                loss,stats=self.loss_fn(pred,batch,metrics=not training)
                if hasattr(self.raw.layer_fusion,'weights'):
                    stats.update({k:v.detach() for k,v in self.raw.layer_fusion.weights().items()})
                else:
                    w6 = self.raw.layer_fusion.layer6_logit.detach().sigmoid()
                    stats['fusion_layer6_weight'] = w6
                    stats['fusion_layer4_weight'] = 1 - w6
                if training:
                    (loss*len(batch['pose'])/total_examples).backward()
                    if sync:
                        grad=torch.nn.utils.clip_grad_norm_([p for p in self.raw.parameters() if p.requires_grad],
                                                             float(self.cfg.TRAIN.CLIP_NORM),error_if_nonfinite=True)
                        self.optimizer.step();self.scheduler.step();self.global_step+=1
                        stats['grad_norm_before_clip']=grad
            mean.add(stats,len(batch['pose']))
            if training and not self.rank and (index%int(self.cfg.TRAIN.SUMMARY_STEP)==0 or index+1==batches):
                LOG.info('TRAIN batch %d/%d | step=%d loss=%.5f local=%.2fmm ramp=%.2f lr=%.3g',
                         index+1,batches,self.global_step,float(loss),float(stats['v5_local_mpjpe_mm']),
                         float(stats['v5_temporal_ramp']),self.optimizer.param_groups[0]['lr'])
        return mean.result(self.device)

    def fit(self):
        for epoch in range(self.start_epoch,int(self.cfg.TRAIN.END_EPOCH)):
            started=time.monotonic()
            if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
            self.loss_fn.set_epoch(epoch)
            self.train_loader.sampler.set_epoch(epoch)
            train=self._run(self.train_loader,True)
            visual={name:self._run(loader,False) for name,loader in self.val_loaders.items()}
            if any(not s for s in visual.values()): raise ValueError('Empty validation source')
            common=set.intersection(*(set(v) for v in visual.values()))
            average={k:sum(v[k] for v in visual.values())/len(visual) for k in common}
            score_key='contact_selection_score' if self.cfg.ROBOT.STAGE=='contact_only' else 'v5_selection_score'
            score=average[score_key]
            is_best=score<self.best
            if is_best:self.best=score
            record={'epoch':epoch+1,'step':self.global_step,'seconds':time.monotonic()-started,
                    'train':train,'validation_by_source':visual,'all_visual_equal_source':average,
                    'selection_score':score,'best':self.best,'is_best':is_best}
            if self.device.type=='cuda':
                memory={'rank':self.rank,'peak_allocated_gib':torch.cuda.max_memory_allocated(self.device)/2**30,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved(self.device)/2**30}
                memory_all=[memory]
                if dist.is_initialized():
                    memory_all=[None]*self.world;dist.all_gather_object(memory_all,memory)
                record['gpu_memory']=memory_all
            if not self.rank:
                if hasattr(self.raw.layer_fusion,'weights'):
                    LOG.info('FEATURE FUSION %s (learned scalar)',
                             {k:round(float(v),6) for k,v in self.raw.layer_fusion.weights().items()})
                else:
                    LOG.info('DECODER FUSION layer4=%.4f layer6=%.4f (learned scalar)',
                             1-float(self.raw.layer_fusion.layer6_logit.sigmoid()),
                             float(self.raw.layer_fusion.layer6_logit.sigmoid()))
                LOG.info('GPU memory peaks GiB: %s',record.get('gpu_memory','CPU'))
                LOG.info('EPOCH %d/%d | %.1fs | best=%.5f%s',epoch+1,self.cfg.TRAIN.END_EPOCH,
                         record['seconds'],self.best,' NEW BEST' if is_best else '')
                log_metrics('TRAIN',train)
                for name,stats in visual.items():log_metrics(name,stats)
                log_metrics('ALL VISUAL (equal-source)',average)
                with (self.logdir/'metrics_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if self.writer:
                    for group,stats in [('train',train),('all_visual',average)]+list(visual.items()):
                        for key,value in stats.items():self.writer.add_scalar(group+'/'+key,value,epoch+1)
                    self.writer.flush()
            self._save(epoch,is_best,record)
            if is_best and not self.rank:
                LOG.info('save the best checkpoint | epoch=%d | selection=%.6f | %s',
                         epoch+1,score,self.logdir/'checkpoints'/'best.pt')
        if self.writer:self.writer.close()
