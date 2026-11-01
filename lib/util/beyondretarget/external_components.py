"""Load user-supplied research components from their documented locations."""

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CAMCALIB_FILES = ('__init__.py', 'model.py', 'cam_utils.py', 'resnet.py')
PLACEMENT_HELP = 'See README.md#spec-and-pare-components for upstream links and file placement.'


def load_camera_regressor():
    directory = ROOT / 'thirdparty/camcalib'
    missing = [name for name in CAMCALIB_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'SPEC/camcalib source is not included in this project. '
            f'Obtain the TRAM-compatible files under the upstream license and place them in {directory}. '
            f'Missing: {", ".join(missing)}. {PLACEMENT_HELP}'
        )
    module = importlib.import_module('thirdparty.camcalib.model')
    model = getattr(module, 'CameraRegressorNetwork', None)
    if not callable(model) or not callable(getattr(model, 'load_ckpt', None)):
        raise ImportError(
            f'{directory / "model.py"} must provide the TRAM-compatible '
            f'CameraRegressorNetwork.load_ckpt API. {PLACEMENT_HELP}'
        )
    return model


def require_camera_resources():
    model = load_camera_regressor()
    checkpoint = ROOT / 'data/pretrain/camcalib_sa_biased_l2.ckpt'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f'Missing SPEC/camcalib checkpoint: {checkpoint}. '
            f'Obtain it under the upstream license before generating world-camera results. {PLACEMENT_HELP}'
        )
    return model, checkpoint


def load_pascal_occlusion():
    source = ROOT / 'thirdparty/pare/coco_occlusion.py'
    if not source.is_file():
        raise FileNotFoundError(
            'PARE/TRAM occlusion source is not included in this project. '
            'Obtain TRAM lib/datasets/coco_occlusion.py under its upstream terms '
            f'and place it at {source}, or keep occlusion disabled. {PLACEMENT_HELP}'
        )
    module = importlib.import_module('thirdparty.pare.coco_occlusion')
    function = getattr(module, 'occlude_with_pascal_objects', None)
    if not callable(function):
        raise ImportError(
            f'{source} must provide occlude_with_pascal_objects. {PLACEMENT_HELP}'
        )
    return function
