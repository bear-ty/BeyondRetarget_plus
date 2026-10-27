"""Read-only, row-exact official TRAM crops; sequence-disjoint v5 splits.

NPZ members are mapped in place (ZIP_STORED), avoiding many-GB copies per
worker. Never enumerate crop directories or silently replace missing images.
"""
from pathlib import Path
import hashlib
import json
import logging
import os
import re
import struct
import zipfile
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, Sampler
from lib.core.constants import IMG_NORM_MEAN, IMG_NORM_STD
from lib.util.beyondretarget.feature_schema import CACHE_FIELDS, validate_shard

SOURCES = {
    'bedlam_vid': ('bedlam_vid.npz','bedlam_30fps/crops',30.,1),
    # Official H36M images are originally 50 Hz, filenames increment by five.
    'h36m_vid': ('h36m_train.npz','h36m/crops',50.,5),
    '3dpw_vid': ('3dpw_vid_train.npz','3dpw/crops',30.,1),
    '3dpw_vid_test': ('3dpw_vid_test.npz','3dpw/crops_test',30.,1),
    'emdb_1': ('emdb_1.npz','emdb/crops_1',30.,1),
}
LOG = logging.getLogger(__name__)


class MappedNPZ:
    def __init__(self, path):
        self.path = str(path)
        self.arrays = {}
        with zipfile.ZipFile(path) as archive, open(path,'rb') as handle:
            self.meta = {}
            for info in archive.infolist():
                if not info.filename.endswith('.npy'):
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(f'{path}: needs uncompressed official NPZ, not a RAM-loaded fallback')
                handle.seek(info.header_offset+26)
                n,e = struct.unpack('<HH',handle.read(4))
                handle.seek(info.header_offset+30+n+e)
                version = np.lib.format.read_magic(handle)
                shape,fortran,dtype = np.lib.format._read_array_header(handle,version)
                if dtype.hasobject:
                    raise ValueError('Object/pickle annotation arrays are not supported')
                self.meta[info.filename[:-4]] = (shape,fortran,dtype,handle.tell())

    def __contains__(self,key):
        return key in self.meta

    def __getitem__(self,key):
        if key not in self.arrays:
            shape,fortran,dtype,offset = self.meta[key]
            self.arrays[key] = np.memmap(self.path,mode='r',dtype=dtype,offset=offset,
                                       shape=shape,order='F' if fortran else 'C')
        return self.arrays[key]

    def __getstate__(self):
        return {'path':self.path,'meta':self.meta,'arrays':{}}


def recording_key(source, imgname):
    if source == 'h36m_vid':
        # Same performance in different cameras must never cross train/val.
        return re.sub(r'\.\d+$','',imgname.rsplit('_',1)[0])
    return str(Path(imgname).parent)


def frame_number(name):
    match = re.search(r'(\d+)\.[^.]+$',str(name))
    if not match:
        raise ValueError(f'Cannot establish chronological identity for {name}')
    return int(match.group(1))


def build_index(db, source, cfg):
    path = Path(db.path)
    n = db.meta['imgname'][0][0]
    limit = int(cfg.DATASET.AUDIT_MAX_ROWS)
    if limit:
        n = min(n,limit)
    key = dict(version=2,path=str(path),size=path.stat().st_size,mtime=path.stat().st_mtime_ns,
               seqlen=int(cfg.DATASET.SEQ_LEN),stride=int(cfg.DATASET.STRIDE),rows=n,
               fraction=float(cfg.DATASET.VAL_FRACTION),seed=int(cfg.DATASET.SPLIT_SEED))
    digest = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(cfg.DATASET.INDEX_DIR)/f'{source}_{digest}.npz'
    if destination.exists():
        with np.load(destination) as cache:
            return cache['starts'],cache['val'],json.loads(str(cache['manifest']))
    LOG.info('INDEX %s: scanning %d annotation rows (no crop directory scan)',source,n)
    length,stride = int(cfg.DATASET.SEQ_LEN),int(cfg.DATASET.STRIDE)
    step = SOURCES[source][3]
    starts,partitions,groups = [],[],{}
    run_start,last_track,last_frame = None,None,None
    last_partition = False
    def close_run(end):
        if run_start is not None:
            entries = list(range(run_start,end-length+1,stride))
            starts.extend(entries)
            partitions.extend([last_partition]*len(entries))
    for left in range(0,n,65536):
        right = min(left+65536,n)
        images = db['imgname'][left:right]
        tracks = db['seqname'][left:right] if 'seqname' in db else None
        valid = np.ones(right-left,dtype=bool)
        if 'invalid' in db: valid &= ~db['invalid'][left:right].astype(bool)
        if 'valid' in db: valid &= db['valid'][left:right]==1
        if 'has_smpl' in db: valid &= db['has_smpl'][left:right].astype(bool)
        if '3dpw' in source:
            center,scale = db['center'][left:right],db['scale'][left:right]
            wh = db['img_shape'][left:right]
            valid &= ((center+scale[:,None]*100)>=0).all(-1)
            valid &= ((center-scale[:,None]*100)<=wh).all(-1)
        for j,image in enumerate(images):
            row = left+j
            frame = frame_number(image)
            track = str(tracks[j]) if tracks is not None else str(image).rsplit('_',1)[0]
            if source == 'emdb_1': track = str(Path(str(image)).parent)
            if not valid[j] or track!=last_track or last_frame is None or frame-last_frame!=step:
                close_run(row)
                run_start = None
            if valid[j] and run_start is None:
                run_start = row
                group = recording_key(source,str(image))
                if group not in groups:
                    value = int(hashlib.sha256(f'{cfg.DATASET.SPLIT_SEED}/{source}/{group}'.encode()).hexdigest()[:12],16)/16**12
                    groups[group] = value < float(cfg.DATASET.VAL_FRACTION)
                last_partition = groups[group]
            last_track,last_frame = track,frame
        if left % (65536*16)==0:
            LOG.info('INDEX %s %d/%d rows',source,right,n)
    close_run(n)
    starts,partitions = np.asarray(starts,dtype=np.int64),np.asarray(partitions,dtype=bool)
    manifest = dict(**key,source=source,clips=len(starts),train_clips=int((~partitions).sum()),
                    validation_clips=int(partitions.sum()),recording_split=groups,
                    fps=SOURCES[source][2]/step,frame_step=step)
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez(f,starts=starts,val=partitions,manifest=json.dumps(manifest))
    os.replace(tmp,destination)
    LOG.info('INDEX %s complete: %d clips; %d train / %d val',source,len(starts),(~partitions).sum(),partitions.sum())
    return starts,partitions,manifest


class RobotVideoDataset(Dataset):
    def __init__(self,cfg,source,split='train'):
        self.cfg,self.source,self.split = cfg,source,split
        filename,folder,original_fps,step = SOURCES[source]
        self.db = MappedNPZ(Path(cfg.DATASET.ROOT)/'dataset_ann'/filename)
        self.crop_dir = Path(cfg.DATASET.ROOT)/folder
        self.starts,val,self.manifest = build_index(self.db,source,cfg)
        if split in ('train','val'):
            self.starts = self.starts[val if split=='val' else ~val]
        if not len(self.starts):
            raise ValueError(f'{source}/{split} has no continuous valid clips')
        self.fps = original_fps/step
        self.occluders = None
        self.augment = split=='train' and bool(cfg.DATASET.PHOTOMETRIC_AUGMENT)
        self.occlusion = split=='train' and bool(cfg.DATASET.OCCLUSION)
        if self.occlusion:
            import joblib
            resource = Path(cfg.DATASET.OCCLUDERS)
            if not resource.is_file():
                raise FileNotFoundError(
                    f'{resource}: run scripts/download_pascal_occluders.sh or '
                    'set DATASET.OCCLUSION=false explicitly'
                )
            self.occluders = joblib.load(resource)
        self.label_db = None
        if cfg.ROBOT.STAGE=='contact_only':
            labelpath = Path(cfg.DATASET.CONTACT_LABEL_DIR)/f'{source}.npz'
            if not cfg.DATASET.CONTACT_LABEL_DIR or not labelpath.is_file():
                raise FileNotFoundError('contact_only requires verified static/contact NPZ sidecars: '+str(labelpath))
            self.label_db = np.load(labelpath,allow_pickle=False)
            if not np.array_equal(self.label_db['imgname'],self.db['imgname']):
                raise ValueError('Contact labels must match ALL annotation rows exactly')
            if 'seqname' in self.db and not np.array_equal(self.label_db['seqname'],self.db['seqname']):
                raise ValueError('Contact label subject/track mismatch')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self,index):
        cv2.setNumThreads(0)
        first = int(self.starts[index])
        n = int(self.cfg.DATASET.SEQ_LEN)
        rows = slice(first,first+n)
        db = self.db
        batch = {key:torch.from_numpy(np.array(db[field][rows],dtype=np.float32))
                 for key,field in [('pose','pose'),('betas','shape'),('center','center'),
                                   ('scale','scale'),('img_focal','img_focal'),('img_center','img_center')]}
        for key,value in batch.items():
            if not torch.isfinite(value).all():
                raise ValueError(f'{self.source}: non-finite {key} at row {first}')
        batch.update(frame_mask=torch.ones(n,dtype=torch.bool),
                     continuous_pair_mask=torch.ones(n-1,dtype=torch.bool),
                     fps=torch.tensor(self.fps,dtype=torch.float32),source=self.source,
                     first_row=torch.tensor(first),frame_ids=torch.tensor([frame_number(x) for x in db['imgname'][rows]]))
        if self.label_db is not None:
            for name in ('static_label','contact_label','static_mask','contact_mask'):
                value = np.asarray(self.label_db[name][rows],dtype=np.float32)
                if value.shape!=(n,2) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
                    raise ValueError(f'Invalid {name} sidecar')
                batch[name] = torch.from_numpy(value.copy())
        cache_dir = str(self.cfg.DATASET.FEATURE_CACHE)
        if cache_dir:
            cache = Path(cache_dir)/self.source/f'{first:08d}.pt'
            if not cache.is_file():
                raise FileNotFoundError(f'Frozen-feature cache incomplete: {cache}')
            frozen = torch.load(cache,map_location='cpu',weights_only=True)
            validate_shard(frozen,first,n)
            for key in CACHE_FIELDS:
                batch['human_'+key] = frozen[key]
            return batch
        colors = np.random.uniform(.8,1.2,3).astype(np.float32) if self.augment else np.ones(3,dtype=np.float32)
        images = []
        for row in range(first,first+n):
            path = self.crop_dir/f'{row:08d}.jpg'
            im = cv2.imread(str(path))
            if im is None or im.shape!=(256,256,3):
                raise FileNotFoundError(f'Missing/corrupt/non-256 official crop: {path}')
            if not im.any():
                raise ValueError(f'Black crop marked valid in annotations: {path}')
            im = im[:,:,::-1].copy()
            if self.occlusion and np.random.random()<.3:
                from lib.datasets.coco_occlusion import occlude_with_pascal_objects
                im = occlude_with_pascal_objects(im,self.occluders)
            im = np.clip(im.astype(np.float32)*colors,0,255)/255.
            im = (im-np.asarray(IMG_NORM_MEAN,dtype=np.float32))/np.asarray(IMG_NORM_STD,dtype=np.float32)
            images.append(torch.from_numpy(im.transpose(2,0,1).copy()))
        batch['img'] = torch.stack(images)
        return batch


class EpochMixtureSampler(Sampler):
    """Same deterministic global draw, disjoint rank shards; source-level weights."""
    def __init__(self,datasets,weights,count,seed,rank=0,world=1):
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum([0]+self.lengths[:-1])
        self.weights = np.asarray(weights,dtype=float)
        if np.any(self.weights<0) or self.weights.sum()<=0:
            raise ValueError('Invalid source sampling weights')
        self.weights /= self.weights.sum()
        self.count = int(count)//world*world
        self.seed,self.rank,self.world,self.epoch = seed,rank,world,0

    def set_epoch(self,epoch): self.epoch=epoch
    def __len__(self): return self.count//self.world
    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        choices = rng.choice(len(self.lengths),size=self.count,p=self.weights)
        indices = np.empty(self.count,dtype=np.int64)
        for i,size in enumerate(self.lengths):
            mask = choices==i
            indices[mask] = self.offsets[i]+rng.integers(size,size=mask.sum())
        return iter(indices[self.rank::self.world].tolist())


def get_robot_loaders(cfg,rank=0,world=1):
    trainsets = [RobotVideoDataset(cfg,s,'train') for s in cfg.DATASET.TRAIN_SOURCES]
    weights = list(cfg.DATASET.SOURCE_WEIGHTS)
    if len(weights)!=len(trainsets): raise ValueError('Source/weight length mismatch')
    sampler = EpochMixtureSampler(trainsets,weights,cfg.DATASET.SAMPLES_PER_EPOCH,cfg.SEED_VALUE,rank,world)
    options = dict(num_workers=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,
                   persistent_workers=cfg.NUM_WORKERS>0)
    if cfg.NUM_WORKERS>0: options['prefetch_factor']=2
    train = DataLoader(ConcatDataset(trainsets),batch_size=cfg.TRAIN.BATCH_SIZE,sampler=sampler,**options)
    val = {}
    for source in cfg.DATASET.VAL_SOURCES:
        split = 'val' if source in cfg.DATASET.TRAIN_SOURCES else 'all'
        db = RobotVideoDataset(cfg,source,split)
        ids = np.arange(len(db))
        cap = int(cfg.DATASET.VAL_MAX_CLIPS)
        if cap and len(ids)>cap:
            ids = np.sort(np.random.default_rng(cfg.DATASET.SPLIT_SEED).choice(ids,cap,replace=False))
        val[source] = DataLoader(Subset(db,ids[rank::world].tolist()),
                                batch_size=cfg.TRAIN.VAL_BATCH_SIZE,shuffle=False,**options)
    return train,val,{'train':{d.source:d.manifest for d in trainsets},
                      'validation':{s:l.dataset.dataset.manifest for s,l in val.items()}}
