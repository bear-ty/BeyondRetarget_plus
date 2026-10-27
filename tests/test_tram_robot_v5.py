import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_rotation_6d,rotation_6d_to_matrix
from lib.core.config import update_cfg
from lib.models.tram_robot import TramRobotAdapter, DecoderLayerFusion
from lib.models.components.pose_transformer import TransformerDecoder
from lib.util.beyondretarget.feature_schema import CACHE_FIELDS
from lib.robot.robot_spec import load_robot_spec
from lib.util.beyondretarget.gvrmr_v5_loss import GvrmrV5Loss
from lib.util.beyondretarget.smpl_supervised_robot_loss import _velocity,_accel
from lib.util.beyondretarget.loss import TramGvrmrV5Loss
from lib.datasets.robot_video_dataset import MappedNPZ,recording_key,EpochMixtureSampler


class RobotV5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(Path(__file__).resolve().parents[1]);torch.set_num_threads(2)
    def setUp(self):torch.manual_seed(42)
    def make(self,n=6):
        loss=GvrmrV5Loss(load_robot_spec('configs/robot_g1.yaml'),{},'cpu',robot_to_target_coord='zup_to_yup')
        root=torch.tensor([1.,0,0,0,1,0]).expand(2,n,6).clone()
        p={'root_pos':torch.zeros(2,n,3),'root_rot_6d':root,'dof':torch.zeros(2,n,29,requires_grad=True)}
        t={'smpl_joints':torch.randn(2,n,24,3)*.2,'root_pos':p['root_pos'].clone(),
           'root_rot_6d':root.clone(),'frame_mask':torch.ones(2,n),'fps':torch.tensor([30.,10.])}
        return loss,p,t
    def test_loss_weights_and_finite_gradients(self):
        loss,p,t=self.make();s=loss(p,t);s['loss'].backward()
        self.assertEqual(loss.weights_v5,dict(position=1.,rotation=.25,velocity=.1,acceleration=.025,dof_smooth=.01,limit=1.))
        self.assertTrue(torch.isfinite(p['dof'].grad).all());self.assertGreater(p['dof'].grad.abs().sum(),0)
        torch.testing.assert_close(s['loss'],sum(s['loss_v5_'+k+'_weighted'] for k in loss.term_names))
    def test_padding_and_short_clips(self):
        for n in (1,2,3):
            loss,p,t=self.make(n);loss(p,t)['loss'].backward();self.assertTrue(torch.isfinite(p['dof'].grad).all())
        loss,p,t=self.make();t['frame_mask'][:,3:]=0
        a=loss(p,t);p['dof'].data[:,3:]=100;t['smpl_joints'][:,3:]=100
        b=loss(p,t)
        for key in loss.term_names:torch.testing.assert_close(a['loss_v5_'+key],b['loss_v5_'+key])
    def test_physical_fps(self):
        value=torch.randn(2,5,3);fps=torch.tensor([30.,10.])
        torch.testing.assert_close(_velocity(value,fps),_velocity(value,fps[:,None].expand(2,5)))
        torch.testing.assert_close(_accel(value,60.),4*_accel(value,30.))
        with self.assertRaises(ValueError):_velocity(value,torch.arange(10.).reshape(2,5))
    def test_root_rotation_invariance(self):
        loss,p,t=self.make();a=loss(p,t)['loss']
        rotation=axis_angle_to_matrix(torch.tensor([.3,.7,-.2]))
        root=matrix_to_rotation_6d(rotation).expand_as(p['root_rot_6d']).clone()
        p['root_rot_6d']=root;t['root_rot_6d']=root;t['smpl_joints']=t['smpl_joints']@rotation.T
        torch.testing.assert_close(a,loss(p,t)['loss'],atol=2e-4,rtol=1e-5)
    def test_warmup_and_validation(self):
        loss,p,t=self.make();a=loss(p,t);loss.set_epoch(4);b=loss(p,t)
        for k in ('velocity','acceleration','dof_smooth'):
            torch.testing.assert_close(5*a['loss_v5_'+k+'_weighted'],b['loss_v5_'+k+'_weighted'])
        loss.eval();loss.set_epoch(0);torch.testing.assert_close(b['loss'],loss(p,t)['loss'])
    def test_gap_masks(self):
        loss,p,t=self.make();t['continuous_pair_mask']=torch.zeros(2,5,dtype=torch.bool)
        s=loss(p,t)
        for k in ('velocity','acceleration','dof_smooth'):self.assertEqual(float(s['loss_v5_'+k]),0.)
    def test_row_rotation_contract(self):
        r=axis_angle_to_matrix(torch.randn(5,3))
        torch.testing.assert_close(rotation_6d_to_matrix(matrix_to_rotation_6d(r)),r)
    def test_head_freeze_and_smpl_supervision(self):
        cfg=update_cfg('configs/config_beyondretarget_g1.yaml')
        model=TramRobotAdapter(cfg,with_human=False);loss=TramGvrmrV5Loss(cfg)
        batch={'human_decoder_layer4':torch.randn(1,16,1024),
               'human_decoder_layer6':torch.randn(1,16,1024),
               'human_smpl_pose6d':torch.tensor([1.,0,0,0,1,0]).repeat(24).expand(1,16,144),
               'human_smpl_betas':torch.zeros(1,16,10),'human_pred_cam':torch.ones(1,16,3),
               'human_root_pos':torch.randn(1,16,3),
               'human_root_rot_6d':torch.tensor([1.,0,0,0,1,0]).expand(1,16,6),
               'pose':torch.zeros(1,16,72),'betas':torch.zeros(1,16,10),
               'frame_mask':torch.ones(1,16,dtype=torch.bool),'fps':torch.tensor([30.]),
               'continuous_pair_mask':torch.ones(1,15,dtype=torch.bool)}
        pred=model(batch,False)
        torch.testing.assert_close(pred['root_pos'],batch['human_root_pos'],rtol=0,atol=0)
        torch.testing.assert_close(pred['root_rot_6d'],batch['human_root_rot_6d'],rtol=0,atol=0)
        value,stats=loss(pred,batch);value.backward()
        self.assertTrue(any(p.grad is not None for p in model.dof_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.contact_head.parameters()))
        self.assertTrue(all(p.grad is None for p in loss.reference.parameters()))
        gate=model.layer_fusion.layer6_logit
        self.assertIsNotNone(gate.grad);self.assertTrue(torch.isfinite(gate.grad))
        self.assertAlmostEqual(float(gate.sigmoid()),.85,places=6)
        self.assertFalse(hasattr(model,'temporal'))
        model.set_stage('contact_only')
        self.assertTrue(all(not p.requires_grad for p in model.dof_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.contact_head.parameters()))
        self.assertFalse(model.layer_fusion.layer6_logit.requires_grad)
    def test_decoder_taps_are_full_blocks_and_do_not_change_output(self):
        model=TransformerDecoder(num_tokens=1,token_dim=1,dim=32,depth=6,heads=2,
                                 mlp_dim=64,dim_head=8,context_dim=24).eval()
        token=torch.zeros(2,1,1);context=torch.randn(2,5,24)
        before=model(token,context=context)
        after,layers=model(token,context=context,return_layers=(4,6))
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        x=model.dropout(model.to_token_embedding(token))+model.pos_embedding[:,:1]
        for i,(sa,ca,ff) in enumerate(model.transformer.layers,1):
            x=sa(x)+x;x=ca(x,context=context)+x;x=ff(x)+x
            if i in (4,6):torch.testing.assert_close(x,layers[i],rtol=0,atol=0)
        self.assertFalse(torch.equal(layers[4],layers[6]))
        with self.assertRaises(ValueError):model(token,context=context,return_layers=(0,))
    def test_fusion_init_and_cache_reuse_across_weights(self):
        fusion=DecoderLayerFusion(.85)
        a,b=torch.randn(2,16,1024),torch.randn(2,16,1024)
        result=fusion(a,b)
        torch.testing.assert_close(result,.15*fusion.norm4(a)+.85*fusion.norm6(b))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(fusion.layer6_logit.grad))
    def test_dof_head_does_not_use_absolute_root_or_camera(self):
        cfg=update_cfg('configs/config_beyondretarget_g1.yaml')
        model=TramRobotAdapter(cfg,with_human=False).eval()
        batch={'human_'+key:torch.randn(1,3,width) for key,width in CACHE_FIELDS.items()}
        before=model(batch,False)['dof']
        batch['human_smpl_pose6d'][...,:6]=torch.randn(1,3,6)
        for key in ('pred_cam','root_pos','root_rot_6d'):batch['human_'+key]=torch.randn_like(batch['human_'+key])*100
        after=model(batch,False)['dof']
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        with self.assertRaises(ValueError):model({'human_context':torch.randn(1,3,1446)},False)
    def test_mmap_no_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.npz';np.savez(path,pose=np.arange(72).reshape(1,72),imgname=np.array(['a_001.jpg']))
            db=MappedNPZ(path);self.assertIsInstance(db['pose'],np.memmap)
            np.testing.assert_array_equal(db['pose'],np.arange(72).reshape(1,72))
    def test_recording_split_and_ddp_shards(self):
        self.assertEqual(recording_key('h36m_vid','S1_Walking_1.54138969_000001.jpg'),
                         recording_key('h36m_vid','S1_Walking_1.55011271_000031.jpg'))
        one=EpochMixtureSampler([range(10),range(6)],[.7,.3],20,42)
        a=EpochMixtureSampler([range(10),range(6)],[.7,.3],20,42,0,2)
        b=EpochMixtureSampler([range(10),range(6)],[.7,.3],20,42,1,2)
        self.assertEqual(list(one),[v for pair in zip(a,b) for v in pair])


if __name__=='__main__':unittest.main(verbosity=2)
