"""Shared six-joint Stage 2. DEFAULT IS CPU PREFLIGHT, not training.

Only --launch constructs an optimizer. The motion-v6 scheduler calls this
after the eight released Stage 1 heads have completed successfully.
3-GPU proposal: CUDA_VISIBLE_DEVICES=2,3,4 torchrun --standalone
--nproc_per_node=3 scripts/train_static_head.py --launch
"""
import argparse, hashlib, json, logging, os, random, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,ConcatDataset,Subset
from lib.core.config import update_cfg
from lib.datasets.static_video_dataset import StaticVideoDataset
from lib.datasets.robot_video_dataset import EpochMixtureSampler
from lib.models.tram_static import TramStaticHead,STATIC_VERSION,JOINT_IDS,JOINT_NAMES
from lib.models.tram_static import build_static_head,forward_static,static_fusion_weights
from lib.util.beyondretarget.feature_schema import CONTEXT_VERSION
from lib.util.beyondretarget.experiment import validate_cache,sha256_file
from lib.util.beyondretarget.static_metrics import masked_bce_sum,metric_counts,summarize_counts


def save_checkpoint(path, payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def get_datasets(settings):
    cfg=update_cfg(settings['base_config'])
    cfg.DATASET.FEATURE_CACHE=settings['feature_cache']
    if cfg.ROBOT.get('ARCHITECTURE','decoder46_v5')!=settings.get('architecture','decoder46_v5'):
        raise ValueError('Static config and base model architecture differ')
    cfg.DATASET.TRAIN_SOURCES=settings['sources'];cfg.DATASET.VAL_SOURCES=settings['sources']
    cache=validate_cache(cfg)
    train=[StaticVideoDataset(cfg,s,'train',settings['label_root'],settings.get('camera_conditioned',False)) for s in settings['sources']]
    val=[StaticVideoDataset(cfg,s,'val',settings['label_root'],settings.get('camera_conditioned',False)) for s in settings['sources']]
    return cache,train,val


def frozen_digest(model):
    frozen=getattr(model,'frozen_motion',None)
    if frozen is None:return None
    if frozen.training or any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
        raise AssertionError('Frozen human replay was made trainable or received gradients')
    digest=hashlib.sha256()
    for key,value in frozen.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/config_beyondretarget_motion_camera_static.yaml')
    p.add_argument('--launch',action='store_true',help='Explicitly authorize actual training')
    p.add_argument('--smoke',action='store_true',help='Two updates plus validation; outputs go under nasdata/audits, never formal experiments')
    p.add_argument('--preflight-report');p.add_argument('--resume')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-per-gpu',type=int)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=p.parse_args();settings=yaml.safe_load(Path(args.config).read_text())
    for arg,key in [('epochs','epochs'),('batch_per_gpu','batch_per_gpu')]:
        if getattr(args,arg) is not None:settings[key]=getattr(args,arg)
    if args.smoke:
        settings['epochs']=1;settings['samples_per_epoch']=2*int(settings['global_batch'])
        if args.resume:raise ValueError('Smoke checks must start fresh')
    torch.set_num_threads(int(os.environ.get('TRAM_CPU_THREADS','2')))
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if not args.launch and world!=1:raise ValueError('Run preflight with plain python, not torchrun')
    cache,trainsets,valsets=get_datasets(settings)
    label_manifests={d.source:d.label_manifest for d in trainsets}
    probe=build_static_head(settings,with_replay=False)
    static_version=probe.static_version
    report=dict(status='preflight_only_no_optimizer',static_version=static_version,
        train_clips={d.source:len(d) for d in trainsets},validation_clips={d.source:len(d) for d in valsets},
        trainable_parameters=sum(p.numel() for p in probe.parameters() if p.requires_grad),
        cache_context=CONTEXT_VERSION,labels=label_manifests,settings=settings,smoke_test=args.smoke)
    if settings.get('camera_conditioned',False):
        from lib.util.beyondretarget.camera_condition import CAMERA_CONDITION_VERSION
        report['camera_condition']=dict(version=CAMERA_CONDITION_VERSION,
            bedlam='true world-to-camera R reconstructed from official row-matched world/incam roots, NOT auxiliary cam_ext',
            h36m='known fixed camera: identity relative rotation',
            window=16,layers=2,width=512,heads=8,ffn=1024,
            uses_gt_body_motion_as_input=False,uses_camera_translation=False,
            label_threshold_mps=.15,postprocess_speed_threshold_mps=.5)
    for d in trainsets+valsets:
        for index in (0,len(d)//2,len(d)-1):
            item=d[index]
            assert item['mask'].any() and item['h6'].shape==(16,1024)
            if getattr(probe,'uses_motion',False):assert item['bbox_info'].shape==(16,3)
            if getattr(probe,'requires_camera',False):
                assert item['camera_rot6d'].shape==(16,6) and torch.isfinite(item['camera_rot6d']).all()
    del probe
    if not args.launch:
        if args.preflight_report:
            dest=Path(args.preflight_report);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='labels'},indent=2))
        return
    batch=int(settings['batch_per_gpu']);global_batch=batch*world
    if global_batch!=settings['global_batch']:
        raise ValueError(f'Global batch {global_batch} != configured {settings["global_batch"]}; explicitly adjust config/LR')
    if int(settings['samples_per_epoch']) % global_batch:
        raise ValueError('Samples per epoch must be divisible by global batch (no short final update)')
    if args.device=='cuda':
        local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:device=torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    seed=settings['seed'];random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed)
    model=build_static_head(settings).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    frozen_before=frozen_digest(model)
    optimizer=torch.optim.AdamW(trainable,lr=settings['lr'],weight_decay=settings['weight_decay'])
    # Same constant-LR policy as v5 Stage 1; no accumulation, no backbone optimizer.
    epoch0=0;steps=0;best=float('inf')
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        previous_settings={k:v for k,v in saved['settings'].items() if k!='epochs'}
        resumed_settings={k:v for k,v in settings.items() if k!='epochs'}
        if previous_settings!=resumed_settings or saved['static_version']!=static_version or saved['label_manifests']!=label_manifests:
            raise ValueError('Resume settings or label provenance changed')
        if int(settings['epochs'])<=int(saved['epoch']):
            raise ValueError('Resume target epochs must exceed the completed checkpoint epoch')
        missing,unexpected=model.load_state_dict(saved['model'],strict=False)
        if any(not k.startswith('frozen_motion.') for k in missing) or unexpected:
            raise ValueError('Static head state mismatch')
        optimizer.load_state_dict(saved['optimizer'])
        epoch0=saved['epoch'];steps=saved['optimizer_steps'];best=saved['best_bce']
        run=Path(args.resume).resolve().parent.parent
    else:
        stamp=[datetime.now().strftime('%Y%m%d_%H%M%S') if rank==0 else None]
        if world>1:dist.broadcast_object_list(stamp,src=0)
        audit_root=Path(os.environ.get('BRPLUS_AUDIT_ROOT', str(Path(settings['output_root'])/'audits')))
        base=audit_root/(settings['experiment']+'_static_smoke') if args.smoke else Path(settings['output_root'])/'experiments'/settings['experiment']
        run=base/'stage2_shared_static'/stamp[0]
    if rank==0:
        run.mkdir(parents=True,exist_ok=bool(args.resume));(run/'checkpoints').mkdir(exist_ok=True)
        if not args.resume:
            (run/'resolved_config.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
            (run/'preflight.json').write_text(json.dumps(report,indent=2))
            for name in ('lib','scripts','configs'):
                shutil.copytree(ROOT/name,run/'code'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            try:
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            except (OSError,subprocess.CalledProcessError):
                commit='unversioned working tree'
            (run/'git_commit.txt').write_text(commit+'\n')
        else:
            # Keep the original run snapshot/config; record extensions separately.
            resume_dir=run/'resumes'/datetime.now().strftime('%Y%m%d_%H%M%S')
            resume_dir.mkdir(parents=True,exist_ok=False)
            (resume_dir/'resolved_config.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
            (resume_dir/'resume.json').write_text(json.dumps(dict(
                checkpoint=str(Path(args.resume).resolve()),checkpoint_sha256=sha256_file(args.resume),
                from_epoch=epoch0,target_epoch=settings['epochs'],optimizer_steps=steps,
                optimizer_restored=True,allowed_setting_changes=['epochs']),indent=2))
            shutil.copy2(__file__,resume_dir/'train_static_head.py')
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run/'tensorboard'))
    if world>1:dist.barrier()
    handlers=[logging.StreamHandler()]
    if rank==0:handlers.append(logging.FileHandler(run/'train.log'))
    logging.basicConfig(level=logging.INFO if rank==0 else logging.WARNING,handlers=handlers,
        format='%(asctime)s | %(message)s',force=True)
    log=logging.getLogger(__name__)
    log.info('STAGE2 STATIC ONLY | version=%s | trainable=%d frozen=%d | fusion=%s',
             static_version,sum(p.numel() for p in trainable),
             sum(p.numel() for p in model.parameters() if not p.requires_grad),static_fusion_weights(model))
    log.info('global batch=%d | per GPU=%d | lr=%g constant | no accumulation | loss=masked BCE only',global_batch,batch,settings['lr'])
    if args.resume:
        log.info('RESUME from epoch=%d step=%d to total epochs=%d | optimizer restored | previous best=%.6f',epoch0,steps,settings['epochs'],best)
    log.info('Human backbone, root and all robot heads are absent from this optimizer; H36M foot/toe labels masked.')
    if settings.get('camera_conditioned',False):
        log.info('CAMERA STATIC | 16-frame native context | 2x512 temporal / 8 heads / FFN1024 | GT BEDLAM relative R + bbox/intrinsics')
        log.info('STATIC LABEL speed<0.15 m/s UNCHANGED; inference double gate speed<0.5 m/s; no GT world-motion input')
    log.info('FROZEN replay SHA256=%s | fresh head=%s',frozen_before,not bool(args.resume))
    sampler=EpochMixtureSampler(trainsets,settings['source_weights'],settings['samples_per_epoch'],seed,rank,world)
    options=dict(num_workers=settings['workers'],pin_memory=device.type=='cuda',persistent_workers=settings['workers']>0)
    loader=DataLoader(ConcatDataset(trainsets),batch_size=batch,sampler=sampler,**options)
    vals={d.source:DataLoader(Subset(d,list(range(rank,len(d),world))),batch_size=batch,**options) for d in valsets}
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None) if world>1 else model
    for epoch in range(epoch0,settings['epochs']):
        sampler.set_epoch(epoch);wrapped.train();counts=torch.zeros(9,6,device=device,dtype=torch.float64);start=time.time()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        grad_sum=0.;camera_grad_sum=0.;camera_moving_tokens=0;camera_tokens=0
        for it,item in enumerate(loader):
            item={k:v.to(device,non_blocking=True) for k,v in item.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and settings['precision']=='bf16'):
                logits=forward_static(wrapped,item)
            loss_sum,n=masked_bce_sum(logits,item['label'],item['mask']);global_n=n.detach().clone()
            if world>1:dist.all_reduce(global_n)
            loss=loss_sum*world/global_n.clamp_min(1)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite static BCE')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(trainable,settings['grad_clip'])
            if not torch.isfinite(grad):raise FloatingPointError('Non-finite static gradient')
            if getattr(model,'requires_camera',False):
                camera_grad_sum+=float(model.camera_embed[0].weight.grad.norm())
                delta=item['camera_rot6d']-model.identity_rotation6d
                camera_moving_tokens+=int((delta.norm(dim=-1)>1e-4).sum())
                camera_tokens+=int(delta.shape[0]*delta.shape[1])
            optimizer.step();steps+=1;grad_sum+=float(grad)
            counts+=metric_counts(logits,item['label'],item['mask'])
            if (it+1)%settings['log_every']==0:
                log.info('epoch=%d step=%d/%d BCE(local)=%.5f grad_norm=%.4f fusion=%s',epoch+1,it+1,len(loader),float(loss_sum/n.clamp_min(1)),float(grad),static_fusion_weights(model))
        if world>1:dist.all_reduce(counts)
        train_summary=summarize_counts(counts);model.eval();validation={}
        with torch.no_grad():
            for source,vloader in vals.items():
                vc=torch.zeros_like(counts)
                for item in vloader:
                    item={k:v.to(device,non_blocking=True) for k,v in item.items()}
                    logits=forward_static(model,item)
                    vc+=metric_counts(logits,item['label'],item['mask'])
                if world>1:dist.all_reduce(vc)
                validation[source]=summarize_counts(vc)
        weights=np.asarray(settings['source_weights']);weights=weights/weights.sum()
        selection=sum(float(w)*validation[s]['bce'] for s,w in zip(settings['sources'],weights))
        if not np.isfinite(selection):raise FloatingPointError('Non-finite validation selection')
        improved=selection<best;best=min(best,selection)
        if frozen_digest(model)!=frozen_before:raise AssertionError('Frozen human weights changed during Stage2')
        if rank==0:
            record=dict(epoch=epoch+1,optimizer_steps=steps,train=train_summary,validation=validation,
                selection_bce=selection,best_bce=best,seconds=time.time()-start,grad_norm_mean=grad_sum/len(loader),
                frozen_replay_unchanged=True,frozen_replay_sha256=frozen_before,
                cuda_peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.,
                cuda_peak_reserved_gb=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0.,
                **static_fusion_weights(model))
            if getattr(model,'requires_camera',False):
                record.update(camera_nonidentity_fraction_rank0=camera_moving_tokens/max(camera_tokens,1),
                              camera_first_weight_grad_norm_mean_rank0=camera_grad_sum/len(loader))
                log.info('CAMERA evidence: nonidentity tokens %.2f%% | first-layer weight grad mean %.6g',
                         100*record['camera_nonidentity_fraction_rank0'],record['camera_first_weight_grad_norm_mean_rank0'])
            log.info('EPOCH %d/%d | train BCE=%.5f | weighted VAL BCE=%.5f best=%.5f | %.1fs',epoch+1,settings['epochs'],train_summary['bce'],selection,best,record['seconds'])
            log.info('INVARIANTS frozen replay unchanged, no backbone grads | GPU0-local peak allocated/reserved %.2f/%.2f GiB',
                     record['cuda_peak_allocated_gb'],record['cuda_peak_reserved_gb'])
            for source,metrics in validation.items():
                log.info('METRICS %s: BCE=%.5f valid=%d',source,metrics['bce'],metrics['valid'])
                for name,entry in metrics['joints'].items():
                    def fmt(value):return 'n/a' if value is None else f'{value:.3f}'
                    def prf(threshold):return '/'.join(fmt(entry[threshold][key]) for key in ('precision','recall','f1'))
                    positive='n/a' if entry['positive_rate'] is None else f'{100*entry["positive_rate"]:.1f}%'
                    log.info('  %-12s n=%-8d pos=%6s | P/R/F1 @.5=%s | @.8=%s',name,entry['valid'],positive,prf('0.5'),prf('0.8'))
                    for threshold in ('0.5','0.8'):
                        for metric,value in entry[threshold].items():
                            if value is not None:writer.add_scalar(f'{source}/{name}/{metric}@{threshold}',value,epoch+1)
                writer.add_scalar(source+'/bce',metrics['bce'],epoch+1)
            writer.add_scalar('train/bce',train_summary['bce'],epoch+1);writer.add_scalar('val/selection',selection,epoch+1)
            for key,value in static_fusion_weights(model).items():writer.add_scalar('train/'+key,value,epoch+1)
            writer.flush()
            with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            head_state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('frozen_motion.')}
            payload=dict(static_version=static_version,context_version=CONTEXT_VERSION,joint_ids=JOINT_IDS,
                model=head_state,optimizer=optimizer.state_dict(),epoch=epoch+1,optimizer_steps=steps,
                best_bce=best,settings=settings,label_manifests=label_manifests,
                human_checkpoint_sha256=cache['contract']['human_checkpoint_sha256'])
            save_checkpoint(run/'checkpoints'/'last.pt',payload)
            if improved:
                save_checkpoint(run/'checkpoints'/'best.pt',payload)
                log.info('save the best checkpoint | epoch=%d | weighted VAL BCE=%.6f | %s',
                         epoch+1,selection,run/'checkpoints'/'best.pt')
        if world>1:dist.barrier()
    if rank==0:writer.close()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
