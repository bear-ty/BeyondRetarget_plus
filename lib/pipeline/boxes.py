"""Pure bounding-box helpers shared by lightweight pipeline entry points."""

import numpy as np


def arrange_boxes(boxes, mode='size', min_size=None):
    """Re-order boxes and optionally filter them by size."""
    if mode == 'left2right':
        cx = (boxes[:, 2] - boxes[:, 0]) / 2 + boxes[:, 0]
        boxes = boxes[np.argsort(cx)]
    elif mode == 'size':
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = w * h
        boxes = boxes[np.argsort(area)[::-1]]
    elif mode == 'conf':
        conf = boxes[:, 4]
        boxes = boxes[np.argsort(conf)[::-1]]
    if min_size is not None:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid = np.stack([w, h]).max(axis=0) > min_size
        boxes = boxes[valid]
    return boxes
