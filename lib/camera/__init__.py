"""Optional camera frontend imports kept lazy for the core adapter path."""


def run_metric_slam(*args, **kwargs):
    from .masked_droid_slam import run_metric_slam as implementation
    return implementation(*args, **kwargs)


def calibrate_intrinsics(*args, **kwargs):
    from .masked_droid_slam import calibrate_intrinsics as implementation
    return implementation(*args, **kwargs)


def align_cam_to_world(*args, **kwargs):
    from .est_gravity import align_cam_to_world as implementation
    return implementation(*args, **kwargs)
