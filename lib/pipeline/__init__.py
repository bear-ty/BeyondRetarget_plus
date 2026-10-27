"""TRAM pipeline entry points with lazy heavy dependencies."""

def video2frames(*args, **kwargs):
    from .tools import video2frames as fn
    return fn(*args, **kwargs)

def detect_segment_track(*args, **kwargs):
    from .tools import detect_segment_track as fn
    return fn(*args, **kwargs)

def visualize_tram(*args, **kwargs):
    from .visualization import visualize_tram as fn
    return fn(*args, **kwargs)
