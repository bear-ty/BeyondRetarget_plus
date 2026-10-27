"""Apply the canonical C-RC1 reprojection and static-foot correction pipeline."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from lib.robot.robot_spec import load_robot_spec
from lib.models.tram_static import load_static_head
from lib.util.beyondretarget.experiment import sha256_file
from lib.util.beyondretarget.inference_policy import add_postprocess_arguments,validate_postprocess_request,resolve_clearance


def process_prediction(output,spec,device='cpu',clearance=None,*,mode='c_rc1',
                       video=None,boxes=None,keypoints=None,confirm_single_shot=False):
    del device
    if mode != 'c_rc1':
        raise ValueError('BeyondRetarget Plus only supports canonical c_rc1 post-processing')
    validate_postprocess_request(mode,video=video,boxes=boxes,keypoints=keypoints,
                                confirm_single_shot=confirm_single_shot)
    clearance=resolve_clearance(mode,clearance)
    if bool(output.get('postprocess_applied',False)):
        raise ValueError('Expected unprocessed raw; do not apply filters/sole offset twice')
    from lib.util.beyondretarget.crc1_postprocess import process_crc1
    return process_crc1(output,spec,video=video,boxes=boxes,keypoints=keypoints,
                        confirm_single_shot=confirm_single_shot,clearance=clearance)


def save_postprocessed(destination,final,report,stages):
    destination=Path(destination)
    if destination.suffix!='.npz':raise ValueError('Output must explicitly end in .npz')
    related=[destination,destination.with_suffix('.diagnostics.json'),destination.with_suffix('.stages.npz')]
    if any(x.exists() for x in related):raise FileExistsError('Choose new output names; do not overwrite existing motion')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation closes the check/write race; publish motion last.
    with related[1].open('x') as handle:handle.write(json.dumps(report,indent=2))
    with related[2].open('xb') as handle:np.savez_compressed(handle,**stages)
    with destination.open('xb') as handle:np.savez_compressed(handle,**final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--static-checkpoint',required=True);p.add_argument('--robot-spec',default='configs/robot_g1.yaml')
    p.add_argument('--device',default='cpu');p.add_argument('--clearance',type=float,default=None,
        help='Human-sole relative offset in metres, default 0; -0.04 adds optional 4cm downward shift (may penetrate)')
    p.add_argument('--video');p.add_argument('--boxes')
    add_postprocess_arguments(p)
    args=p.parse_args()
    validate_postprocess_request(args.postprocess_mode,video=args.video,boxes=args.boxes,
                                keypoints=args.keypoints,confirm_single_shot=args.confirm_single_shot)
    if Path(args.input).resolve()==Path(args.output).resolve():raise ValueError('Keep raw output intact')
    with np.load(args.input,allow_pickle=False) as d:output={k:d[k] for k in d.files}
    load_static_head(args.static_checkpoint,'cpu',str(output['human_checkpoint_sha256']))
    if str(output.get('static_checkpoint_sha256'))!=sha256_file(args.static_checkpoint):
        raise ValueError('Static probabilities and checkpoint SHA mismatch')
    final,report,stages=process_prediction(output,load_robot_spec(args.robot_spec),args.device,args.clearance,
        mode=args.postprocess_mode,video=args.video,boxes=args.boxes,keypoints=args.keypoints,
        confirm_single_shot=args.confirm_single_shot)
    report.setdefault('sources',{}).update(raw=str(Path(args.input).resolve()),robot_spec=str(Path(args.robot_spec).resolve()),
                                          static_checkpoint=str(Path(args.static_checkpoint).resolve()))
    report.setdefault('sha256',{}).update(raw=sha256_file(args.input),robot_spec=sha256_file(args.robot_spec),
                                         static_checkpoint=sha256_file(args.static_checkpoint))
    save_postprocessed(args.output,final,report,stages)
    print(json.dumps({'output':args.output,'method':report['method'],'frames':len(final['dof']),'clips':len(report['clips'])},indent=2))


if __name__=='__main__':main()
