"""Crop TRAM datasets into the image layout expected by the training caches."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import ROOT


SAVE_DIRS = {
    'h36m_vid': ROOT + '/h36m/crops',
    '3dpw_vid': ROOT + '/3dpw/crops',
    'bedlam_vid': ROOT + '/bedlam_30fps/crops',
    '3dpw_vid_test': ROOT + '/3dpw/crops_test',
    'emdb_1': ROOT + '/emdb/crops_1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=sorted(SAVE_DIRS), default=['bedlam_vid'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.crop_size < 1:
        parser.error('batch-size/workers/crop-size must be positive (workers may be zero)')

    import cv2
    import numpy as np
    import random
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from lib.datasets.base_dataset import BaseDataset

    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for dataset_name in args.datasets:
        print(f'Processing (crop) {dataset_name} ...')
        dataset = BaseDataset(dataset_name, is_train=True, crop_size=args.crop_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        image_dir = Path(SAVE_DIRS[dataset_name])
        image_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in tqdm(loader):
            for image in batch['img'].numpy():
                if not cv2.imwrite(str(image_dir / f'{count:08d}.jpg'), image):
                    raise IOError(f'Could not write {image_dir / f"{count:08d}.jpg"}')
                count += 1
        print(f'{dataset_name}: wrote {count} crops to {image_dir}')


if __name__ == '__main__':
    main()
