import os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import numpy as np
from lib.util.beyondretarget.crc1 import image_support_weights,joint_xy_solve,validate_motion
from lib.util.beyondretarget.crc1_image_evidence import evidence_from_pairs,project
from lib.util.beyondretarget.crc1 import CRC1Config
from lib.util.beyondretarget.crc1_postprocess import validate_inputs


class CandidateTests(unittest.TestCase):
    def fixture(self,fps=30,n=61,physical_speed=0.,camera_shift=0.,zoom=1.):
        p=np.full((n,6),.9);kp=np.zeros((n,17,3));kp[:,:,2]=.99
        boxes=np.tile([0.,0.,200.,200.],(n,1));hs=[];cumulative=np.eye(3)
        for t in range(n):
            for j in [15,16]:
                point=np.array([50.+(j-15)*60.+physical_speed*200.*t/fps,180.])
                kp[t,j,:2]=project(point,cumulative)
            boxes[t]*=zoom**t
            h=np.array([[zoom,0.,camera_shift],[0.,zoom,0.],[0.,0.,1.]])
            if t<n-1:hs.append(h)
            cumulative=h@cumulative
        return p,kp,boxes,np.asarray(hs),np.ones(n-1,bool)

    def test_stationary_with_camera_pan(self):
        p,k,b,h,v=self.fixture(camera_shift=4.)
        e=evidence_from_pairs(k,b,h,v,30)
        self.assertLess(np.max(e['speed_bodyheight_s'][4:-4]),1e-10)
        self.assertGreater((image_support_weights(p,e,30)>0).sum(),80)

    def test_stationary_with_camera_zoom(self):
        p,k,b,h,v=self.fixture(zoom=1.015,camera_shift=2.)
        e=evidence_from_pairs(k,b,h,v,30)
        self.assertLess(np.max(e['speed_bodyheight_s'][4:-4]),1e-10)

    def test_fast_sliding_rejected_despite_high_probability(self):
        p,k,b,h,v=self.fixture(physical_speed=.2,camera_shift=4.)
        e=evidence_from_pairs(k,b,h,v,30)
        self.assertEqual((image_support_weights(p,e,30)>0).sum(),0)

    def test_low_head_probability_rejected(self):
        p,k,b,h,v=self.fixture();p[:]=.49
        self.assertEqual((image_support_weights(p,evidence_from_pairs(k,b,h,v,30),30)>0).sum(),0)

    def test_bad_background_fails_closed(self):
        p,k,b,h,v=self.fixture();v[:]=False
        self.assertEqual((image_support_weights(p,evidence_from_pairs(k,b,h,v,30),30)>0).sum(),0)

    def test_occluded_ankles_rejected(self):
        p,k,b,h,v=self.fixture();k[:,[15,16],2]=.1
        self.assertEqual((image_support_weights(p,evidence_from_pairs(k,b,h,v,30),30)>0).sum(),0)

    def test_real_fps_and_resolution_scaling(self):
        for fps in [15,30,60]:
            p,k,b,h,v=self.fixture(fps=fps,physical_speed=.05,n=2*fps+1)
            e=evidence_from_pairs(k,b,h,v,fps)
            self.assertAlmostEqual(float(e['speed_bodyheight_s'][fps,0]),.05,places=8)
            k[:,:,:2]*=2;b*=2
            e2=evidence_from_pairs(k,b,h,v,fps)
            self.assertAlmostEqual(float(e2['speed_bodyheight_s'][fps,0]),.05,places=8)

    def test_short_false_episode_removed_and_last_row_invalid(self):
        p=np.full((20,6),.9)
        e=dict(speed_bodyheight_s=np.zeros((20,2)),uncertainty_bodyheight_s=np.zeros((20,2)),evidence=np.zeros((20,2),bool))
        e['evidence'][5:7]=True;e['evidence'][-1]=True
        self.assertEqual(np.count_nonzero(image_support_weights(p,e,30)),0)

    def test_joint_solver_preserves_z_and_constant_velocity(self):
        r=np.column_stack([np.arange(50)*.1,np.arange(50)*.3,np.sin(np.arange(50))])
        result=joint_xy_solve(r,np.zeros((50,2,3)),np.zeros((50,2)),30)
        np.testing.assert_array_equal(result[:,2],r[:,2])
        np.testing.assert_allclose(result,r,atol=1e-9)

    def test_joint_solver_support_reduces_slip(self):
        n=90;r=np.column_stack([np.arange(n)*.01,np.zeros((n,2))])
        out=joint_xy_solve(r,np.zeros((n,2,3)),np.ones((n,2)),30)
        self.assertLess(np.abs(np.diff(out[:,0])).mean(),np.abs(np.diff(r[:,0])).mean()*.5)

    def test_root_translation_equivariance(self):
        rng=np.random.default_rng(2);r=rng.normal(size=(50,3));f=rng.normal(size=(50,2,3))*.1;p=np.ones((50,2))
        shift=np.array([11.,-7.,3.])
        a=joint_xy_solve(r,f,p,30);b=joint_xy_solve(r+shift,f,p,30)
        np.testing.assert_allclose(b,a+shift,atol=1e-8)

    def test_track_gaps_rejected(self):
        ids=np.array([0,1,4]);raw=dict(frame=ids,static_prob6=np.ones((3,6)),root_pos_world_zup=np.zeros((3,3)))
        motion=dict(root_trans=np.zeros((3,3)),root_rot_quat=np.tile([1.,0,0,0],(3,1)),dof=np.zeros((3,29)))
        with self.assertRaisesRegex(ValueError,'Split temporal gaps'):
            validate_motion(raw,motion,np.zeros((3,3)),30)

    def test_invalid_probabilities_rejected(self):
        e=dict(speed_bodyheight_s=np.zeros((10,2)),uncertainty_bodyheight_s=np.zeros((10,2)),evidence=np.ones((10,2),bool))
        with self.assertRaises(ValueError):image_support_weights(np.full((10,6),np.nan),e,30)

    def test_single_frame_does_not_crash_solver(self):
        r=np.array([[1.,2.,3.]])
        np.testing.assert_array_equal(joint_xy_solve(r,np.zeros((1,2,3)),np.zeros((1,2)),30),r)

    def test_single_frame_has_no_support(self):
        e=evidence_from_pairs(np.ones((1,17,3)),np.array([[0.,0.,100.,200.]]),
                              np.empty((0,3,3)),np.empty(0,bool),30.)
        self.assertEqual(np.count_nonzero(image_support_weights(np.ones((1,6)),e,30.)),0)

    def test_config_validation(self):
        for kw in [{'ik_cap_degrees':0.},{'probability_min':1.},{'root_position_sigma_m':np.nan},
                   {'constraint_weight_min':.5},{'ik_iterations':0}]:
            with self.assertRaises(ValueError):CRC1Config(**kw)

    def test_network_probability_is_not_overwritten(self):
        p,k,b,h,v=self.fixture();p[:]=.6;original=p.copy()
        w=image_support_weights(p,evidence_from_pairs(k,b,h,v,30),30)
        np.testing.assert_array_equal(p,original)
        self.assertEqual(float(w.max()),.95)
        np.testing.assert_array_equal(w[:,[0,2,4,5]],0)


class InputContractTests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        n=8
        spec=SimpleNamespace(name='g1',dof=29,model_dof_order=['j'+str(i) for i in range(29)])
        raw=dict(frame=np.arange(n),fps=np.asarray(30.),robot_name=np.asarray('g1'),
            postprocess_applied=np.asarray(False),static_head_trained=np.asarray(True),
            root_pos_world_zup=np.tile([0.,0.,3.],(n,1)),root_quat_world_zup_wxyz=np.tile([1.,0,0,0],(n,1)),
            dof=np.zeros((n,29)),pred_pose=np.zeros((n,144)),pred_shape=np.zeros((n,10)),
            pred_trans=np.tile([0.,0.,3.],(n,1)),static_prob6=np.ones((n,6))*.9,
            camera_to_world_zup_R=np.tile(np.eye(3),(n,1,1)),camera_to_world_zup_T=np.zeros((n,3)),
            joint_names=np.asarray(spec.model_dof_order))
        boxes=dict(boxes=np.tile([0.,0.,100.,200.],(n,1)),frame=np.arange(n),img_focal=np.asarray(500.),img_center=np.array([100.,100.]))
        points=dict(keypoints=np.zeros((n,17,3)),frame=np.arange(n),fps=np.asarray(30.))
        points['keypoints'][:,:,2]=.9
        return raw,spec,boxes,points

    def test_valid_inputs_and_no_silent_second_filter(self):
        args=self.fixture();self.assertEqual(validate_inputs(*args,confirm_single_shot=True),(8,30.))
        args[0]['postprocess_applied']=np.asarray(True)
        with self.assertRaisesRegex(ValueError,'unprocessed raw'):validate_inputs(*args,confirm_single_shot=True)

    def test_frame_identity_and_gap_checks(self):
        for target in ['boxes','points','raw']:
            raw,spec,boxes,points=self.fixture()
            {'boxes':boxes,'points':points,'raw':raw}[target]['frame'][-1]+=2
            with self.assertRaises(ValueError):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)

    def test_intrinsics_must_match_raw(self):
        raw,spec,boxes,points=self.fixture();raw['img_focal']=np.asarray(450.)
        with self.assertRaisesRegex(ValueError,'Intrinsics differ'):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)

    def test_vitpose_scores_may_exceed_one_without_clipping(self):
        raw,spec,boxes,points=self.fixture();points['keypoints'][:,:,2]=1.04
        validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)
        self.assertTrue((points['keypoints'][:,:,2]==1.04).all())
        points['keypoints'][0,0,2]=-1
        with self.assertRaises(ValueError):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)

    def test_camera_tracking_and_joint_order(self):
        raw,spec,boxes,points=self.fixture();raw['camera_tracking_valid']=np.zeros(8,bool)
        with self.assertRaisesRegex(ValueError,'Unreliable camera'):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)
        raw.pop('camera_tracking_valid');raw['joint_names']=raw['joint_names'][::-1]
        with self.assertRaisesRegex(ValueError,'joint names'):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)

    def test_no_head_or_invalid_geometry_is_rejected(self):
        for key,value in [('static_head_trained',False),('static_prob6',np.ones((8,6))*1.1),
                          ('root_quat_world_zup_wxyz',np.zeros((8,4))),('camera_to_world_zup_R',np.zeros((8,3,3)))]:
            raw,spec,boxes,points=self.fixture();raw[key]=value
            with self.assertRaises(ValueError):validate_inputs(raw,spec,boxes,points,confirm_single_shot=True)

    def test_output_preservation_and_stage_serialization(self):
        import tempfile
        from scripts.postprocess_beyondretarget import save_postprocessed
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'result.npz'
            save_postprocessed(dest,{'dof':np.zeros((2,29))},{'method':'test'}, {'crc1_image__evidence':np.zeros((2,2),bool)})
            before=dest.read_bytes()
            with self.assertRaises(FileExistsError):save_postprocessed(dest,{}, {}, {})
            self.assertEqual(before,dest.read_bytes())
            with np.load(dest.with_suffix('.stages.npz'),allow_pickle=False) as data:
                self.assertEqual(data['crc1_image__evidence'].shape,(2,2))

    def test_missing_2d_evidence_refuses_fit(self):
        from lib.util.beyondretarget.crc1_reprojection import fit_translation
        raw,_,boxes,points=self.fixture();points['keypoints'][:,:,2]=0
        with self.assertRaisesRegex(ValueError,'Insufficient reliable'):
            fit_translation(raw,points['keypoints'],boxes)

    def test_reprojection_refuses_invalid_initial_depth(self):
        from unittest.mock import patch
        from lib.util.beyondretarget.crc1_reprojection import fit_translation
        raw,_,boxes,points=self.fixture();raw['root_pos_world_zup'][:]=0
        with patch('lib.util.beyondretarget.crc1_reprojection.body_joints',return_value=np.zeros((8,24,3))):
            with self.assertRaisesRegex(ValueError,'behind/too close'):
                fit_translation(raw,points['keypoints'],boxes)


if __name__=='__main__':unittest.main(verbosity=2)
