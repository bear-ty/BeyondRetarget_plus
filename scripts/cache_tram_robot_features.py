"""Precompute frozen TRAM outputs once, then train all robot heads without ViT.

Supports torchrun rank sharding and restart (validated existing shards skipped).
Never caches augmented samples. A partial cache is NOT marked training-ready.
"""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
if not os.environ.get('PYTHONNOUSERSITE'):
    os.environ['PYTHONNOUSERSITE']='1'
    os.execv(sys.executable,[sys.executable,'-s',*sys.argv])
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from lib.core.config import update_cfg
from lib.datasets.robot_video_dataset import RobotVideoDataset
from lib.models.tram_robot import TramRobotAdapter,load_human_weights
from lib.util.beyondretarget.experiment import cache_contract
from lib.beyondretarget_trainer import move_batch
from lib.util.beyondretarget.feature_schema import validate_shard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg',default='configs/config_beyondretarget_g1.yaml')
    p.add_argument('--output',default='outputs/feature_cache/vimo_decoder46_16f_v2')
    p.add_argument('--sources',default='bedlam_vid,h36m_vid,3dpw_vid')
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--max-clips',type=int,default=0,help='Diagnostic extraction only; incomplete manifest')
    p.add_argument('--audit-max-rows',type=int,default=0)
    args=p.parse_args();cfg=update_cfg(args.cfg)
    cfg.DATASET.FEATURE_CACHE='';cfg.DATASET.PHOTOMETRIC_AUGMENT=False;cfg.DATASET.OCCLUSION=False
    cfg.DATASET.AUDIT_MAX_ROWS=args.audit_max_rows
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device('cuda',local) if args.device=='cuda' else torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    # Workers infer independently; long NAS/source imbalance must not trip a
    # short collective timeout while another rank is still doing useful work.
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(hours=6))
    torch.set_num_threads(int(os.environ.get('TRAM_CPU_THREADS','4')))
    logging.basicConfig(level=logging.INFO,format=f'%(asctime)s rank={rank} %(message)s')
    log=logging.getLogger(__name__)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    contract=cache_contract(cfg)
    contract['extraction_device']=device.type
    # CPU vs CUDA extraction precision is metadata; mixed extraction devices
    # in the same cache are rejected. Training may use either device.
    path=root/'manifest.json'
    old={}
    had_manifest=path.exists()
    if had_manifest:
        old=json.loads(path.read_text())
        if old['contract']!=contract:raise ValueError('Refusing incompatible preexisting feature cache')
    sources=old.get('sources',{})
    if not had_manifest and any(root.glob('*/*.pt')):
        raise ValueError('Unversioned existing cache shards: choose a new cache directory')
    if not rank:
        initial={'contract':contract,'sources':sources,'complete':False}
        tmp=root/'manifest.json.tmp';tmp.write_text(json.dumps(initial,indent=2));os.replace(tmp,path)
    if world>1:dist.barrier()
    model=TramRobotAdapter(cfg).to(device).eval()
    missing,unexpected=load_human_weights(model,cfg.MODEL.CHECKPOINT)
    log.info('Loaded official weights; missing=%s unexpected=%s',missing,unexpected)
    for source in args.sources.split(','):
        if rank==0:db=RobotVideoDataset(cfg,source,'all')
        if world>1:dist.barrier()
        if rank!=0:db=RobotVideoDataset(cfg,source,'all')
        (root/source).mkdir(exist_ok=True)
        count=min(len(db),args.max_clips) if args.max_clips else len(db)
        pending=[]
        for index in range(rank,count,world):
            first=int(db.starts[index]);destination=root/source/f'{first:08d}.pt'
            if destination.exists():
                cached=torch.load(destination,map_location='cpu',weights_only=True)
                validate_shard(cached,first)
            else:pending.append(index)
        loader=DataLoader(Subset(db,pending),batch_size=args.batch_size,num_workers=args.num_workers,
                          pin_memory=device.type=='cuda',shuffle=False)
        log.info('%s clips=%d this rank pending=%d',source,len(db),len(pending))
        started=time.monotonic()
        completed=0
        def progress(done):
            seconds=time.monotonic()-started
            value={'rank':rank,'world_size':world,'source':source,'source_total_clips':len(db),
                   'rank_pending_at_start':len(pending),'rank_newly_cached':completed,
                   'elapsed_seconds':seconds,'clips_per_second':completed/max(seconds,1e-6),
                   'source_rank_complete':done,'updated_unix':time.time()}
            target=root/f'progress_rank{rank}.json'
            temporary=target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,target)
        progress(False)
        with torch.no_grad():
            for step,batch in enumerate(loader):
                batch=move_batch(batch,device)
                _,frozen=model._human_forward(batch)
                for i,first in enumerate(batch['first_row'].tolist()):
                    payload={k:v[i].float().cpu() for k,v in frozen.items()}
                    if not all(torch.isfinite(v).all() for v in payload.values()):
                        raise FloatingPointError(f'Non-finite frozen features: {source}/{first}')
                    payload.update(first_row=first,length=16)
                    validate_shard(payload,first)
                    destination=root/source/f'{first:08d}.pt'
                    tmp=destination.with_suffix(f'.{os.getpid()}.tmp')
                    torch.save(payload,tmp);os.replace(tmp,destination)
                completed+=len(batch['first_row'])
                if step%50==0:
                    progress(False)
                    log.info('%s %d/%d batches; %.2f clips/s',source,step+1,len(loader),
                             completed/max(time.monotonic()-started,1e-6))
        progress(True)
        metadata={k:db.manifest[k] for k in ('path','size','mtime','rows')}
        metadata['full_rows']=db.db.meta['imgname'][0][0]
        sources[source]={'annotation':metadata,'clips':len(db),'cached_clips':count}
        if world>1:dist.barrier()
        if not rank:
            manifest={'contract':contract,'sources':sources,'complete':False}
            tmp=root/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,path)
    if not rank:
        manifest['complete']=(not args.max_clips and not args.audit_max_rows and
                              all(s['cached_clips']==s['clips'] and s['annotation']['rows']==s['annotation']['full_rows']
                                  for s in sources.values()))
        tmp=root/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,path)
        log.info('Feature extraction done. complete=%s manifest=%s',manifest['complete'],path)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
