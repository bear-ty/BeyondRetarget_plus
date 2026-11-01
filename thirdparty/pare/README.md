# User-supplied PARE/TRAM occlusion

This project does not have permission to redistribute this component, so its
implementation is not included. Obtain TRAM's
[`lib/datasets/coco_occlusion.py`](https://github.com/yufu-wang/tram/blob/main/lib/datasets/coco_occlusion.py)
under the applicable [PARE terms](https://github.com/mkocabas/PARE/blob/master/LICENSE)
and [synthetic-occlusion terms](https://github.com/isarandi/synthetic-occlusion).
Retain the license notices and put the file at
`thirdparty/pare/coco_occlusion.py`; no path registration is needed.

For the Pascal occluder resource, follow its upstream terms and run
`bash scripts/download_pascal_occluders.sh` from the repository root. This
places the data at `data/pascal_occluders.pkl`.

Only optional synthetic-occlusion training needs these files. Public Plus
configs disable it, and default training, inference, and evaluation work
without it. Use augmentation without a frozen-feature cache and verify with
the commands below (the upstream helper imports `loguru`):

```bash
python -m pip install loguru
python scripts/check_installation.py --require-occlusion
```

User-supplied source is Git-ignored.
