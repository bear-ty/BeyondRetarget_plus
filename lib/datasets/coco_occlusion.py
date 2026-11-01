"""Forward optional Pascal augmentation to the user's upstream implementation."""

from lib.util.beyondretarget.external_components import load_pascal_occlusion


def occlude_with_pascal_objects(im, occluders):
    return load_pascal_occlusion()(im, occluders)
