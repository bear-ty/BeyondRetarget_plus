"""Immutable data/weights contracts and experiment provenance."""
import hashlib
import json
from pathlib import Path
from lib.util.beyondretarget.feature_schema import CONTEXT_VERSION, CACHE_FIELDS


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def cache_contract(cfg):
    root=Path(__file__).resolve().parents[3]
    source_files = ['lib/models/hmr_vimo.py', 'lib/models/modules.py',
                    'lib/models/components/pose_transformer.py', 'lib/models/vit.py',
                    'lib/util/beyondretarget/feature_schema.py', 'lib/datasets/robot_video_dataset.py']
    return {'context_version':CONTEXT_VERSION,'sequence_length':int(cfg.DATASET.SEQ_LEN),
            'stride':int(cfg.DATASET.STRIDE),'human_checkpoint_sha256':sha256_file(cfg.MODEL.CHECKPOINT),
            'human_forward_sha256':sha256_file(root/'lib/models/hmr_vimo.py'),
            'frozen_source_sha256':{name:sha256_file(root/name) for name in source_files},
            'fields':CACHE_FIELDS,'feature_tap':'1-based layers 4 and 6, full block post-FFN residual query',
            'fusion':'not cached; both raw tokens stored independently',
            'smpl_pose_convention':'post-motion HMR2 column-6D, 24 joints including root',
            'root_rotation_convention':'pytorch3d row-6D from official rotation matrix',
            'preprocessing':'official_256crop_imagenet_no_geometric_no_photometric_no_occlusion_v1',
            'human_mode':'eval_frozen','precision':'cuda_fp16_backbone_else_cpu_fp32'}


def validate_cache(cfg):
    if not cfg.DATASET.FEATURE_CACHE:return None
    if cfg.DATASET.PHOTOMETRIC_AUGMENT or cfg.DATASET.OCCLUSION:
        raise ValueError('Fixed feature caches cannot be combined with random image augmentation')
    path=Path(cfg.DATASET.FEATURE_CACHE)/'manifest.json'
    if not path.is_file():raise FileNotFoundError(f'Missing cache provenance: {path}')
    manifest=json.loads(path.read_text())
    expected=cache_contract(cfg)
    stored={k:v for k,v in manifest['contract'].items() if k!='extraction_device'}
    if stored!=expected:
        raise ValueError('Feature cache contract differs from current checkpoint/code/preprocessing')
    if not manifest.get('complete'):
        raise ValueError('Partial feature cache cannot be used for formal training; finish extraction first')
    for source in set(cfg.DATASET.TRAIN_SOURCES)|set(cfg.DATASET.VAL_SOURCES):
        if source not in manifest['sources']:
            raise ValueError('Feature cache missing source '+source)
        metadata=manifest['sources'][source]['annotation']
        path=Path(metadata['path'])
        from lib.datasets.robot_video_dataset import SOURCES
        current=Path(cfg.DATASET.ROOT)/'dataset_ann'/SOURCES[source][0]
        if current.resolve()!=path.resolve():
            raise ValueError('Configured annotations differ from feature cache: '+source)
        if path.stat().st_mtime_ns!=metadata['mtime'] or path.stat().st_size!=metadata['size']:
            raise ValueError('Annotation changed since feature extraction: '+source)
        if metadata['rows']<metadata['full_rows']:
            raise ValueError('Audit-only feature cache cannot be used for formal training')
    return manifest
