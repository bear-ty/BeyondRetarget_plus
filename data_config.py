"""Dataset layout and runtime paths.

Set ``TRAM_DATA_ROOT`` and ``TRAM_ANN_ROOT`` when the dataset is stored
outside the repository. Defaults stay under ``data/datasets``.
"""
from os.path import join
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'datasets'
ROOT = os.environ.get('TRAM_DATA_ROOT', str(DEFAULT_DATA_ROOT))
DATASET_NPZ_PATH = os.environ.get('TRAM_ANN_ROOT', join(ROOT, 'dataset_ann'))

H36M_ROOT         = join(ROOT, 'h36m')
PW3D_ROOT         = join(ROOT, '3dpw')
BEDLAM_ROOT       = join(ROOT, 'bedlam_30fps')
EMDB_ROOT         = join(ROOT, 'emdb')

# Path to test/train npz files
DATASET_FILES = [ {
                   'emdb_1': join(DATASET_NPZ_PATH , 'emdb_1.npz'),
                   '3dpw_vid_test': join(DATASET_NPZ_PATH , '3dpw_vid_test.npz'),
                  },

                  {
                   '3dpw_vid': join(DATASET_NPZ_PATH , '3dpw_vid_train.npz'),
                   'h36m_vid': join(DATASET_NPZ_PATH , 'h36m_train.npz'),
                   'bedlam_vid': join(DATASET_NPZ_PATH , 'bedlam_vid.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m_vid': H36M_ROOT,
                   '3dpw_vid': PW3D_ROOT,
                   'bedlam_vid': BEDLAM_ROOT,
                   'emdb_1': EMDB_ROOT,
                   '3dpw_vid_test': PW3D_ROOT,
                }


PASCAL_OCCLUDERS = os.environ.get(
    'BRPLUS_PASCAL_OCCLUDERS',
    str(PROJECT_ROOT / 'data' / 'pascal_occluders.pkl'),
)
SMPL_MODEL_DIR = os.environ.get('BRPLUS_SMPL_ROOT', str(PROJECT_ROOT / 'data' / 'smpl'))
JOINT_REGRESSOR_TRAIN_EXTRA = str(Path(SMPL_MODEL_DIR) / 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = str(Path(SMPL_MODEL_DIR) / 'J_regressor_h36m.npy')
SMPL_MEAN_PARAMS = str(Path(SMPL_MODEL_DIR) / 'smpl_mean_params.npz')
