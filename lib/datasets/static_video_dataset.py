"""Static-only loader: unchanged Stage 1 windows/cache, row-exact masked labels."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.datasets.robot_video_dataset import MappedNPZ, SOURCES, build_index
from lib.util.beyondretarget.static_labels import verify_label_manifest
from lib.util.beyondretarget.feature_schema import validate_shard


class StaticVideoDataset(Dataset):
    def __init__(self, cfg, source, split, label_root, camera_conditioned=False):
        self.source = source
        self.camera_conditioned = bool(camera_conditioned)
        self.use_motion = cfg.ROBOT.get('ARCHITECTURE','decoder46_v5') == 'decoder6_motion_v6'
        self.length = int(cfg.DATASET.SEQ_LEN)
        self.db = MappedNPZ(Path(cfg.DATASET.ROOT)/'dataset_ann'/SOURCES[source][0])
        if self.camera_conditioned and (source not in ('bedlam_vid','h36m_vid')
                or (source=='bedlam_vid' and 'cam_ext' not in self.db)):
            raise ValueError('No verified camera condition for '+source)
        starts, val, self.split_manifest = build_index(self.db, source, cfg)
        if split not in ('train', 'val'): raise ValueError('Use recording-disjoint train/val')
        self.starts = starts[val if split=='val' else ~val]
        self.label_dir = Path(label_root)/source
        self.label_manifest = verify_label_manifest(self.label_dir, self.db.path)
        if self.camera_conditioned and source=='bedlam_vid':
            import json
            from lib.util.beyondretarget.experiment import sha256_file
            if self.label_manifest.get('position_policy')!='native_smpl_true_world_from_gvhmr_v2':
                raise ValueError('BEDLAM cam_ext is not the true moving-camera transform. Build verified true-camera v2 labels first.')
            path=self.label_dir/'camera_manifest.json'
            if sha256_file(path)!=self.label_manifest['camera_manifest_sha256']:
                raise ValueError('Camera and static label provenance mismatch')
            manifest=json.loads(path.read_text())
            if not manifest['complete'] or manifest['contract']['annotation']!=self.label_manifest['annotation']:
                raise ValueError('Incomplete or misaligned true camera sidecar')
            if sha256_file(self.label_dir/'camera_R.npy')!=manifest['files']['camera_R']:
                raise ValueError('True camera rotation SHA mismatch')
        self.cache_dir = Path(cfg.DATASET.FEATURE_CACHE)/source
        self.arrays = {}
        # Only exclude clips with zero supervision, never based on predicted quality.
        mask = np.load(self.label_dir/'mask.npy', mmap_mode='r')
        keep = np.zeros(len(self.starts), bool)
        for k in range(0,len(self.starts),8192):
            rows=self.starts[k:k+8192,None]+np.arange(self.length)[None]
            keep[k:k+8192]=mask[rows].any(axis=(1,2))
        self.starts=self.starts[keep]
        if not len(self.starts): raise ValueError('No supervised static windows: '+source+'/'+split)

    def __len__(self): return len(self.starts)

    def __getstate__(self):
        state=self.__dict__.copy();state['arrays']={};return state

    def __getitem__(self, index):
        first=int(self.starts[index]);rows=slice(first,first+self.length)
        if not self.arrays:
            self.arrays={k:np.load(self.label_dir/(k+'.npy'),mmap_mode='r') for k in ('label','mask')}
            if self.camera_conditioned and self.source=='bedlam_vid':
                self.arrays['camera_R']=np.load(self.label_dir/'camera_R.npy',mmap_mode='r')
        frozen=torch.load(self.cache_dir/f'{first:08d}.pt',map_location='cpu',weights_only=True)
        validate_shard(frozen,first,self.length)
        item = dict(h6=frozen['decoder_layer6'],
            label=torch.from_numpy(self.arrays['label'][rows].astype(np.float32)),
            mask=torch.from_numpy(self.arrays['mask'][rows].copy()))
        if self.use_motion:
            from lib.models.tram_motion_robot import bbox_information
            metadata={key:torch.from_numpy(np.asarray(self.db[key][rows],dtype=np.float32).copy())
                      for key in ('center','scale','img_focal','img_center')}
            item['bbox_info']=bbox_information(metadata)
        else:
            item['h4']=frozen['decoder_layer4']
        if self.camera_conditioned:
            from lib.util.beyondretarget.camera_condition import camera_relative_6d
            rotation=(np.asarray(self.arrays['camera_R'][rows]) if self.source=='bedlam_vid'
                      else np.broadcast_to(np.eye(3),(self.length,3,3)))
            item['camera_rot6d']=torch.from_numpy(camera_relative_6d(rotation))
        return item
