"""Output contract for the canonical Plus inference pipeline."""
from dataclasses import dataclass
from pathlib import Path


def add_postprocess_arguments(parser, mode_option='--mode'):
    parser.add_argument(mode_option, dest='postprocess_mode', choices=('c_rc1',), default='c_rc1',
                        help='Canonical C-RC1 image-supported static XY/IK with translation reprojection.')
    parser.add_argument('--keypoints', help='Frozen COCO17 pixel detections [T,17,3], .npy or .npz with keypoints/frame/fps')
    parser.add_argument('--confirm-single-shot', action='store_true',
                        help='Caller confirms cuts, missing intervals and different target identities were split')


def validate_postprocess_request(mode, *, video=None, boxes=None, keypoints=None, confirm_single_shot=False):
    if mode != 'c_rc1':
        raise ValueError('Unknown postprocess mode: ' + str(mode))
    if not video or not boxes or not keypoints or not confirm_single_shot:
        raise ValueError('C-RC1 requires --video, --boxes, --keypoints and --confirm-single-shot.')
    for name, path in [('video', video), ('boxes', boxes), ('keypoints', keypoints)]:
        if not Path(path).is_file():
            raise FileNotFoundError(name + ': ' + str(path))


def resolve_clearance(mode, clearance=None):
    """Metres relative to predicted human soles, NOT absolute floor height.

    Keep the former zero offset by default. --clearance -0.04 explicitly adds
    the reviewed optional 4 cm downward shift; it can cause floor penetration.
    """
    if mode != 'c_rc1':
        raise ValueError('Unknown postprocess mode: ' + str(mode))
    import math
    value=0. if clearance is None else float(clearance)
    if not math.isfinite(value):raise ValueError('Clearance must be finite')
    return value


@dataclass(frozen=True)
class OutputPlan:
    raw: Path
    final: Path | None

    @property
    def primary(self):
        return self.final if self.final is not None else self.raw

    @property
    def targets(self):
        result=[self.raw]
        if self.final is not None:
            result += [self.final, self.final.with_suffix('.diagnostics.json'), self.final.with_suffix('.stages.npz')]
        return result


def plan_outputs(output, *, raw_only=False, raw_output=None):
    """Plan a final output and an optional explicitly named raw sidecar."""
    output=Path(output)
    if raw_only:
        if raw_output:
            raise ValueError('--raw-only cannot be combined with --raw-output')
        plan=OutputPlan(output,None)
    else:
        plan=OutputPlan(Path(raw_output) if raw_output else output.with_name(output.stem+'.raw.npz'),output)
    if plan.raw.suffix!='.npz' or (plan.final is not None and plan.final.suffix!='.npz'):
        raise ValueError('Motion output filenames must explicitly end in .npz')
    resolved=[path.resolve() for path in plan.targets]
    if len(set(resolved))!=len(resolved):raise ValueError('Raw/final/diagnostic/stage output paths overlap')
    return plan


def validate_request(plan, static_checkpoint, camera_trajectory, smooth_alpha=0.):
    if not 0<=smooth_alpha<=1:raise ValueError('smooth-alpha must be in [0,1]')
    if plan.final is not None:
        # ``static_checkpoint`` is optional when the robot checkpoint is the
        # released unified Plus bundle: infer_beyondretarget loads the embedded
        # shared static head from that payload.  Keep the camera trajectory
        # mandatory for C-RC1, while allowing either an explicit standalone
        # static checkpoint or the merged bundle path.
        if not camera_trajectory:
            raise ValueError('Default output is fully postprocessed. Supply a merged/static checkpoint and '
                '--camera-trajectory; use --raw-only explicitly for raw inference.')
        if smooth_alpha:
            raise ValueError('Canonical postprocessing already filters; do not pre-apply EMA')
    for path in plan.targets:
        if path.exists():raise FileExistsError('Preserve existing output; choose new filenames: '+str(path))
