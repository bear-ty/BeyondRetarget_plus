"""Train a frozen-human BeyondRetarget robot adapter."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
if not os.environ.get('PYTHONNOUSERSITE'):
    os.environ['PYTHONNOUSERSITE']='1'
    os.execv(sys.executable,[sys.executable,'-s',*sys.argv])
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lib.core.config import update_cfg
from lib.datasets.robot_video_dataset import get_robot_loaders
from lib.models.tram_robot import TramRobotAdapter,load_human_weights,load_tram_robot_checkpoint,CONTEXT_VERSION
from lib.models.tram_motion_robot import build_robot_model
from lib.util.beyondretarget.loss import TramGvrmrV5Loss
from lib.util.beyondretarget.experiment import sha256_file,validate_cache
from lib.beyondretarget_trainer import RobotTrainer


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg',default='configs/config_beyondretarget_g1.yaml')
    p.add_argument('--exp-name');p.add_argument('--device')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--val-batch-size',type=int)
    p.add_argument('--effective-batch-size',type=int);p.add_argument('--num-workers',type=int)
    p.add_argument('--train-sources');p.add_argument('--val-sources')
    p.add_argument('--feature-cache');p.add_argument('--resume');p.add_argument('--from-checkpoint')
    p.add_argument('--robot-spec');p.add_argument('--stage',choices=['head_only','contact_only'])
    p.add_argument('--max-train-batches',type=int);p.add_argument('--max-val-batches',type=int)
    p.add_argument('--audit-max-rows',type=int)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--smoke',action='store_true',help='One epoch, two train/one val batches; NOT formal training')
    args=p.parse_args()
    config_path=Path(args.resume).resolve().parent.parent/'resolved_config.yaml' if args.resume else Path(args.cfg)
    cfg=update_cfg(str(config_path))
    for arg,section,key in [('exp_name',None,'EXP_NAME'),('device',None,'DEVICE'),('epochs','TRAIN','END_EPOCH'),
        ('batch_size','TRAIN','BATCH_SIZE'),('effective_batch_size','TRAIN','EFFECTIVE_BATCH_SIZE'),
        ('num_workers',None,'NUM_WORKERS'),('feature_cache','DATASET','FEATURE_CACHE'),('resume','TRAIN','RESUME'),
        ('from_checkpoint','TRAIN','FROM_CHECKPOINT'),('robot_spec','ROBOT','SPEC'),('stage','ROBOT','STAGE'),
        ('max_train_batches','TRAIN','MAX_TRAIN_BATCHES'),('max_val_batches','TRAIN','MAX_VAL_BATCHES'),
        ('audit_max_rows','DATASET','AUDIT_MAX_ROWS')]:
        value=getattr(args,arg)
        if value is not None:(cfg if section is None else cfg[section])[key]=value
    if args.train_sources:
        cfg.DATASET.TRAIN_SOURCES=args.train_sources.split(',')
        base={'bedlam_vid':.65,'h36m_vid':.2,'3dpw_vid':.15}
        cfg.DATASET.SOURCE_WEIGHTS=[base[x] for x in cfg.DATASET.TRAIN_SOURCES]
    if args.val_sources:cfg.DATASET.VAL_SOURCES=args.val_sources.split(',')
    if args.resume and args.from_checkpoint is None:
        # The original stage2 run records its initialization checkpoint; resume
        # restores full state and must not initialize from that checkpoint again.
        cfg.TRAIN.FROM_CHECKPOINT=''
    if args.val_batch_size is not None:
        cfg.TRAIN.VAL_BATCH_SIZE=args.val_batch_size
    elif cfg.DATASET.FEATURE_CACHE:
        cfg.TRAIN.VAL_BATCH_SIZE=cfg.TRAIN.BATCH_SIZE
    if args.smoke:
        cfg.TRAIN.END_EPOCH=1;cfg.TRAIN.MAX_TRAIN_BATCHES=2;cfg.TRAIN.MAX_VAL_BATCHES=1
        cfg.TRAIN.EFFECTIVE_BATCH_SIZE=cfg.TRAIN.BATCH_SIZE*int(os.environ.get('WORLD_SIZE','1'))
        cfg.NUM_WORKERS=0;cfg.EXP_NAME=cfg.EXP_NAME+'_smoke'
    elif cfg.DATASET.AUDIT_MAX_ROWS or cfg.TRAIN.MAX_TRAIN_BATCHES or cfg.TRAIN.MAX_VAL_BATCHES:
        cfg.EXP_NAME=cfg.EXP_NAME+'_limited_audit'
    if cfg.TRAIN.RESUME and cfg.TRAIN.FROM_CHECKPOINT:
        raise ValueError('Choose resume OR stage initialization, not both')
    if cfg.ROBOT.STAGE=='contact_only' and not (cfg.TRAIN.FROM_CHECKPOINT or cfg.TRAIN.RESUME):
        raise ValueError('contact_only must start from a trained robot head')
    return args,cfg


def main():
    args,cfg=arguments()
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if cfg.DEVICE.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
        if world==1 and torch.device(cfg.DEVICE).index is not None:
            local_rank=torch.device(cfg.DEVICE).index
        torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    else:device=torch.device(cfg.DEVICE)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    torch.set_num_threads(int(os.environ.get('TRAM_CPU_THREADS','4')))
    random.seed(cfg.SEED_VALUE+rank);np.random.seed(cfg.SEED_VALUE+rank);torch.manual_seed(cfg.SEED_VALUE+rank)
    if cfg.TRAIN.RESUME:
        run=Path(cfg.TRAIN.RESUME).resolve().parent.parent
    else:
        stage='stage1' if cfg.ROBOT.STAGE=='head_only' else 'stage2'
        from lib.robot.robot_spec import load_robot_spec
        spec=load_robot_spec(cfg.ROBOT.SPEC)
        # Epoch defaults come from the selected config (60 for robot configs).
        # Never silently override a non-G1 head or a resumed historical run.
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        if world>1:
            shared=[stamp if rank==0 else None];dist.broadcast_object_list(shared,src=0);stamp=shared[0]
        testing=args.smoke or args.preflight or cfg.DATASET.AUDIT_MAX_ROWS or cfg.TRAIN.MAX_TRAIN_BATCHES or cfg.TRAIN.MAX_VAL_BATCHES
        audit_root = Path(os.environ.get('BRPLUS_AUDIT_ROOT', str(Path(cfg.OUTPUT_DIR) / 'audits')))
        experiment_root=audit_root if testing else Path(cfg.OUTPUT_DIR)/'experiments'
        run=(experiment_root/cfg.EXP_NAME/f'{stage}_{spec.name}_{cfg.ROBOT.STAGE}'/f'{stamp}_{spec.name}')
    cfg.LOGDIR=str(run)
    if not rank:run.mkdir(parents=True,exist_ok=bool(cfg.TRAIN.RESUME))
    if world>1:dist.barrier()
    handlers=[logging.StreamHandler()]
    if not rank:handlers.append(logging.FileHandler(run/'train.log',encoding='utf-8'))
    logging.basicConfig(level=logging.INFO if not rank else logging.WARNING,
                        format='%(asctime)s | %(levelname)s | %(message)s',handlers=handlers,force=True)
    log=logging.getLogger(__name__)
    log.info('TRAM robot v5 | device=%s rank=%d/%d stage=%s',device,rank,world,cfg.ROBOT.STAGE)
    log.info('No human auxiliary loss: ALL human/root modules are frozen. Robot v5 weights=%s',dict(cfg.ROBOT_LOSS))
    cache=validate_cache(cfg)
    # Build cache once before other ranks open the indices.
    if rank==0:loaders=get_robot_loaders(cfg,rank,world)
    if world>1:dist.barrier()
    if rank!=0:loaders=get_robot_loaders(cfg,rank,world)
    train,val,manifests=loaders
    if not rank:
        record_dir=(run/'resume_records'/datetime.now().strftime('%Y%m%d_%H%M%S')) if cfg.TRAIN.RESUME else run
        record_dir.mkdir(parents=True,exist_ok=True)
        (record_dir/'resolved_config.yaml').write_text(cfg.dump())
        (record_dir/'data_split_manifest.json').write_text(json.dumps(manifests,indent=2))
    if args.preflight:
        for dataset in train.dataset.datasets:
            item=dataset[0];log.info('PREFLIGHT %s clips=%d pose=%s fps=%.1f input=%s',
                dataset.source,len(dataset),tuple(item['pose'].shape),float(item['fps']),
                'cached decoder4/6 tokens' if 'human_decoder_layer4' in item else tuple(item['img'].shape))
        log.info('Preflight passed; no optimizer or training launched.')
        if world>1:dist.destroy_process_group()
        return
    model=build_robot_model(cfg,with_human=not bool(cfg.DATASET.FEATURE_CACHE))
    if model.human is not None:
        missing,unexpected=load_human_weights(model,cfg.MODEL.CHECKPOINT)
        log.info('Official human weights loaded: missing nonparameter keys=%s unexpected=%s',missing,unexpected)
    model.to(device)
    if cfg.TRAIN.FROM_CHECKPOINT:load_tram_robot_checkpoint(model,cfg.TRAIN.FROM_CHECKPOINT)
    model.set_stage(cfg.ROBOT.STAGE)
    params=[p for p in model.parameters() if p.requires_grad]
    log.info('Parameters: trainable=%d total=%d; root never optimized',sum(p.numel() for p in params),sum(p.numel() for p in model.parameters()))
    loss=TramGvrmrV5Loss(cfg,device=device).to(device)
    optimizer=torch.optim.AdamW(params,lr=float(cfg.TRAIN.LR),weight_decay=float(cfg.TRAIN.WD))
    spec=model.robot_spec
    try:
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        git_diff=subprocess.run(['git','diff'],cwd=ROOT,capture_output=True,check=True).stdout
    except (OSError,subprocess.CalledProcessError):
        git_commit='unversioned working tree';git_diff=''
    provenance={'context_version':getattr(model,'context_version',CONTEXT_VERSION),
                'cache_context_version':CONTEXT_VERSION,'architecture':cfg.ROBOT.get('ARCHITECTURE','decoder46_v5'),
                'git_commit':git_commit,
                'robot_spec_sha256':sha256_file(cfg.ROBOT.SPEC),
                'robot_asset_sha256':sha256_file(spec.xml_path),
                'human_checkpoint':str(Path(cfg.MODEL.CHECKPOINT).resolve()),
                'human_checkpoint_sha256':cache['contract']['human_checkpoint_sha256'] if cache else sha256_file(cfg.MODEL.CHECKPOINT),
                'gvrmr_v5_source_commit':'f016326ff27977008a5dc61198806417df57ea26',
                'root_basis':'smpl_parameter_axes_v2','source_fps':{k:v['fps'] for k,v in manifests['train'].items()},
                'smoke_test':args.smoke,'world_size':world}
    log.info('Architecture=%s | model context=%s | immutable cache context=%s',
             provenance['architecture'],provenance['context_version'],CONTEXT_VERSION)
    if getattr(model,'frozen_motion',None) is not None:
        assert not any(p.requires_grad for p in model.frozen_motion.parameters())
        log.info('Motion replay: official 16-frame FP32/eval/no_grad, frozen params=%d; no ViT replay, no h4 input',
                 sum(p.numel() for p in model.frozen_motion.parameters()))
    if not rank:
        (record_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
        snap=record_dir/'run_snapshot'
        for folder in ('lib','configs','scripts'):
            shutil.copytree(ROOT/folder,snap/folder,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'data_config.py',snap/'data_config.py')
        assets=snap/'robot_assets';assets.mkdir(exist_ok=True)
        shutil.copy2(spec.xml_path,assets/Path(spec.xml_path).name)
        (snap/'git_diff.patch').write_text(git_diff)
    if world>1:model=DistributedDataParallel(model,device_ids=[local_rank] if device.type=='cuda' else None,broadcast_buffers=False)
    RobotTrainer(cfg,model,loss,(train,val),optimizer,provenance).fit()
    log.info('Finished: %s',run)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
