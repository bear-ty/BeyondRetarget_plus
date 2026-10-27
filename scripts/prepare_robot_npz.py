"""Merge/validate TRAM annotations with robot supervision.

The robot annotation file must contain ``robot_dof`` [N,D] and may contain
``robot_root_pos``, ``robot_root_rot_6d``, ``robot_shared_joints``,
``robot_valid``, ``robot_contact`` and ``fps``.  If both files contain
``imgname``, rows are aligned by image name; otherwise equal-length row order
is required.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

OPTIONAL = ('robot_root_pos','robot_root_rot_6d','robot_shared_joints','robot_valid','robot_contact','fps','frame_mask','continuous_pair_mask')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--human-ann', required=True)
    ap.add_argument('--robot-ann', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--check-only', action='store_true')
    args=ap.parse_args()
    human=dict(np.load(args.human_ann,allow_pickle=True)); robot=dict(np.load(args.robot_ann,allow_pickle=True))
    if 'robot_dof' not in robot: raise KeyError('robot annotation requires robot_dof [N,D]')
    hn=len(human['imgname']) if 'imgname' in human else len(next(iter(human.values())))
    rn=len(robot['imgname']) if 'imgname' in robot else len(robot['robot_dof'])
    if 'imgname' in human and 'imgname' in robot:
        hnames=[x.decode() if isinstance(x,bytes) else str(x) for x in human['imgname']]
        rnames=[x.decode() if isinstance(x,bytes) else str(x) for x in robot['imgname']]
        index={name:i for i,name in enumerate(rnames)}
        missing=[name for name in hnames if name not in index]
        if missing: raise ValueError(f'{len(missing)} human rows have no robot label, first={missing[:3]}')
        order=np.asarray([index[name] for name in hnames],dtype=np.int64)
    else:
        if hn!=rn: raise ValueError(f'row count mismatch human={hn} robot={rn}')
        order=np.arange(hn)
    out=dict(human)
    for key in ('robot_dof',)+OPTIONAL:
        if key in robot and key!='imgname': out[key]=np.asarray(robot[key])[order]
    out.setdefault('robot_valid',np.ones(hn,dtype=np.float32))
    if np.asarray(out['robot_dof']).shape[0]!=hn: raise ValueError('robot_dof length mismatch after alignment')
    if args.check_only:
        print('OK:',hn,'frames; robot_dof=',np.asarray(out['robot_dof']).shape)
        return
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**out)
    print('Wrote',args.output,'frames=',hn,'robot_dof=',np.asarray(out['robot_dof']).shape)

if __name__=='__main__': main()
