"""MuJoCo playback: robot-follow camera, fixed initial viewing direction.

No root rotation/trajectory deformation. A constant display XYZ translation
places the predicted HUMAN reference floor at z=0; the saved motion is untouched.
"""
import argparse,json,os,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
# EGL enumerates physical devices independently of CUDA's logical numbering.
# In single-GPU jobs keep rendering on the selected physical CUDA device too.
if os.environ.get('CUDA_VISIBLE_DEVICES','').isdigit():
    os.environ['MUJOCO_EGL_DEVICE_ID']=os.environ['CUDA_VISIBLE_DEVICES']
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import cv2,numpy as np,mujoco as mj,imageio_ffmpeg,yaml
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from lib.robot.robot_spec import load_robot_spec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',required=True);p.add_argument('--output',required=True)
    p.add_argument('--spec',default='configs/robot_g1.yaml');p.add_argument('--elevation',type=float,default=-30)
    p.add_argument('--reference-final',required=True,help='Common human-floor and camera reference for raw/final')
    p.add_argument('--fit-other',required=True,help='Fit the union of raw and final motion geometry')
    p.add_argument('--motion-format',choices=['tram','gvrmr'],default='tram')
    p.add_argument('--raw-ground-shift',type=float,default=0.,help='GVRMR raw display-only constant Z offset; never changes saved motion')
    p.add_argument('--label',default='TRAM robot G1')
    p.add_argument('--resolution',type=int,default=960)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--follow',dest='follow',action='store_true',help='Follow each robot; common raw/final angle and zoom (default)')
    mode.add_argument('--fixed-camera',dest='follow',action='store_false',help='Fixed full-trajectory overview')
    p.set_defaults(follow=True)
    args=p.parse_args();dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    def load_motion(path):
        value=dict(np.load(path))
        if args.motion_format=='gvrmr':
            value['postprocess_applied']=np.asarray(Path(path).resolve()==Path(args.reference_final).resolve())
            if not bool(value['postprocess_applied']):
                value['root_trans']=value['root_trans'].copy()
                value['root_trans'][:,2]+=args.raw_ground_shift
            return value
        if not bool(value['postprocess_applied']):
            value['root_trans']=value['root_pos_world_zup']
            value['root_rot_quat']=value['root_quat_world_zup_wxyz']
        return value
    d=load_motion(args.motion);other=load_motion(args.fit_other);reference=load_motion(args.reference_final)
    assert bool(reference['postprocess_applied'])
    spec=load_robot_spec(args.spec);n=len(d['dof']);fps=float(d['fps'])
    scene=ET.parse(spec.xml_path).getroot()
    xml_directory=Path(spec.xml_path).resolve().parent
    geometry_options=yaml.safe_load((ROOT/'configs/static_geometry.yaml').read_text()).get(spec.name,{})
    compiler=scene.find('compiler')
    if compiler is not None:
        compiler.set('meshdir',str(Path(geometry_options['meshdir']).resolve()) if geometry_options.get('meshdir')
                     else str((xml_directory/compiler.get('meshdir','.')).resolve()))
    for a in scene.findall('asset'):
        for tex in a.findall('texture'):
            if tex.get('file'):tex.set('file',str((xml_directory/tex.get('file')).resolve()))
    for world in scene.findall('worldbody'):
        for geom in list(world.findall('geom')):
            if geom.get('type')=='plane':world.remove(geom)
    world=scene.find('worldbody')
    asset=scene.find('asset')
    if asset is None:asset=ET.SubElement(scene,'asset')
    ET.SubElement(asset,'texture',name='review_checker',type='2d',builtin='checker',width='512',height='512',rgb1='.22 .26 .31',rgb2='.32 .36 .41')
    ET.SubElement(asset,'material',name='review_ground',texture='review_checker',texrepeat='2 2',texuniform='true',reflectance='0')
    ET.SubElement(world,'geom',name='demo_floor',type='plane',pos='0 0 0',size='200 200 .01',material='review_ground',contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='2 -2 4',diffuse='.8 .8 .8')
    model=mj.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'));data=mj.MjData(model)
    size=args.resolution
    model.vis.global_.offwidth=size;model.vis.global_.offheight=size
    model.vis.headlight.active=1;model.vis.headlight.ambient[:]=[.4,.4,.4]
    renderer=mj.Renderer(model,size,size);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
    quat=d['root_rot_quat'];rot=Rotation.from_quat(reference['root_rot_quat'][:,[1,2,3,0]]).as_matrix()
    forward=rot[:min(15,n),:,0].mean(0)
    cam.azimuth=float(np.degrees(np.arctan2(forward[1],forward[0])));cam.elevation=args.elevation
    if args.motion_format=='gvrmr':
        floor=0.;fallback=False
    else:
        human=reference['human_sole_height_world_zup'];prob=reference['static_prob6'][:,[1,3]]
        vals=human[prob>.8]
        fallback=not len(vals)
        if fallback:vals=human.reshape(-1)
        floor=float(np.sort(vals)[:max(1,int(np.ceil(len(vals)*.05)))].mean())
    origin=np.array([*reference['root_trans'][0,:2],floor]);root=d['root_trans']-origin
    def pose(i,motion=d):
        data.qpos[:]=model.qpos0;data.qpos[:3]=motion['root_trans'][i]-origin;data.qpos[3:7]=motion['root_rot_quat'][i]
        data.qpos[7+np.asarray(spec.model_to_xml_dof)]=motion['dof'][i];mj.mj_forward(model,data)
    gids=np.flatnonzero(model.geom_bodyid>0);radius=model.geom_rbound[gids,None]
    low=np.full(3,np.inf);high=np.full(3,-np.inf);camera_path=None
    for motion_index,motion in enumerate((d,other)):
        assert len(motion['dof'])==n
        # Do not include the raw/final TRAJECTORY separation in zoom: each
        # panel follows its own pelvis. Match angle and zoom, NOT camera XYZ.
        centers=motion['root_trans'].astype(np.float64)-origin
        window=min(21,n if n%2 else n-1)
        if window>=5:centers=savgol_filter(centers,window,2,axis=0)
        if motion_index==0:camera_path=centers
        for i in range(len(motion['dof'])):
            pose(i,motion);offset=centers[i] if args.follow else np.zeros(3)
            low=np.minimum(low,(data.geom_xpos[gids]-radius-offset).min(0))
            high=np.maximum(high,(data.geom_xpos[gids]+radius-offset).max(0))
    corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
    if not args.follow:camera_path=np.tile((low+high)/2,(n,1))
    # Fit at the origin for follow mode, making paired zoom numerically equal.
    cam.lookat[:]=np.zeros(3) if args.follow else camera_path[0]
    cam.distance=3.;renderer.update_scene(data,cam)
    gl=renderer.scene.camera[0];f=np.array(gl.forward);u=np.array(gl.up);right=np.cross(f,u)
    relative=corners-cam.lookat;tan=np.tan(np.deg2rad(model.vis.global_.fovy)/2)
    cam.distance=max(2.8,float(np.max(np.maximum(abs(relative@right),abs(relative@u))/(tan*.75)-relative@f))+.2)
    renderer.update_scene(data,cam);gl=renderer.scene.camera[0];delta=corners-np.array(gl.pos);depth=delta@f
    ratio=float(np.max(np.maximum(abs(delta@right),abs(delta@u))/(depth*tan)))
    assert depth.min()>0 and ratio<.85
    # G1 XML fixes stat.extent=0.8: default far=40m can clip a wide
    # full-trajectory view even when the projected XY bounds fit perfectly.
    model.vis.map.zfar=max(float(model.vis.map.zfar),float((depth.max()+20.)/model.stat.extent))
    assert depth.min()>model.vis.map.znear*model.stat.extent
    assert depth.max()<model.vis.map.zfar*model.stat.extent
    from OpenGL import GL
    gpu=GL.glGetString(GL.GL_RENDERER).decode()
    dest.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{size}x{size}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(dest)],stdin=subprocess.PIPE)
    try:
        for i in range(n):
            pose(i);cam.lookat[:]=camera_path[i];renderer.update_scene(data,cam);img=renderer.render().copy()
            cv2.rectangle(img,(0,0),(size,48),(240,240,240),-1)
            stage='POSTPROCESSED' if bool(d['postprocess_applied']) else 'RAW'
            view='FOLLOW' if args.follow else 'FIXED WORLD'
            cv2.putText(img,f'{args.label} | {stage} | {view} | {i+1}/{n}',(12,31),cv2.FONT_HERSHEY_SIMPLEX,.48 if size<900 else .65,(0,0,0),1)
            proc.stdin.write(img.tobytes())
        proc.stdin.close();assert proc.wait()==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    dest.with_suffix('.render.json').write_text(json.dumps(dict(frames=n,fps=fps,gpu=gpu,
        egl_device_id=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        camera_azimuth=float(cam.azimuth),camera_elevation=float(cam.elevation),camera_distance=float(cam.distance),
        camera_mode='per_robot_smoothed_follow' if args.follow else 'fixed_global',
        camera_path=camera_path.tolist() if args.follow else None,
        all_frames_in_view=True,max_projected_half_screen_ratio=ratio,
        depth_range_m=[float(depth.min()),float(depth.max())],far_clip_m=float(model.vis.map.zfar*model.stat.extent),
        motion_orientation_preserved=True,display_translation=(-origin).tolist(),
        floor_policy='GVRMR final z=0 floor; raw constant display shift only' if args.motion_format=='gvrmr' else 'constant predicted HUMAN sole low5% mean at static feet; no motion correction',
        ground='world-fixed checker, matching A_prefilter_worldgate',raw_display_z_shift=args.raw_ground_shift if args.motion_format=='gvrmr' else None,
        floor_selection_fallback_all_frames=fallback,physics='kinematic playback; no contact simulation'),indent=2))
    print('RENDER COMPLETE',dest,flush=True)

if __name__=='__main__':main()
