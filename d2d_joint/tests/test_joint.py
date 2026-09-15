import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from d2d_joint.physics import PhaseBank,suites
from d2d_joint.environment import Environment,Config,action_id,FRAME_DIM,drift_path,metrics
from d2d_joint.controllers import Classical
from d2d_joint.ppo import ActorCritic,Policy,gae,train
from d2d_joint.design import DesignSearch
from d2d_sizing.circuit import NOMINAL,file_hash


def bank():
    phase=np.linspace(-.5,1.5,513);bits=np.tile(np.arange(64)%2,(4,2,1));signed=np.clip(.35-2*np.abs(phase-.5),-.5,.5)
    voltage=.5+(2*bits[:,:,:,None]-1)*signed
    return PhaseBank(phase,voltage,bits,[.25]*4,[0.]*4,suites()[0],NOMINAL)


class JointTests(unittest.TestCase):
    def test_selected_design_and_policy_integrity(self):
        from d2d_joint.run import validate_selected
        from types import SimpleNamespace
        from d2d_sizing.circuit import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy';path.write_bytes(b'fixed version')
            bank=SimpleNamespace(x=np.array([1.,2.]))
            row=dict(x=[1.,2.],policy_seed=0,policy_sha256=file_hash(path))
            validate_selected(row,bank,path)
            with self.assertRaises(ValueError):validate_selected(dict(row,x=[1.,3.]),bank,path)
            path.write_bytes(b'different version')
            with self.assertRaises(ValueError):validate_selected(row,bank,path)

    def test_phase_and_time_persist_without_wrap(self):
        e=Environment(bank(),np.zeros((64,2)))
        e.step(action_id([4,-4],False));self.assertEqual(e.phase.tolist(),[36,28])
        e.step(action_id([0,0],False));self.assertEqual(e.phase.tolist(),[36,28]);self.assertEqual(e.t,2)
        e.phase[:]=64;e.step(action_id([4,4],False));self.assertEqual(e.phase.tolist(),[64,64])
        np.testing.assert_array_equal(e.x,NOMINAL)

    def test_hidden_drift_reward_and_info_do_not_enter_observation(self):
        a=Environment(bank(),np.zeros((64,2)));b=Environment(bank(),np.full((64,2),.3))
        np.testing.assert_array_equal(a.observation(),b.observation())
        oa,ra,_,ia=a.step(action_id([0,0],False));ob,rb,_,ib=b.step(action_id([0,0],False))
        self.assertNotEqual(ra,rb);self.assertNotEqual(ia['margins'],ib['margins'])
        np.testing.assert_array_equal(oa,ob)
        self.assertEqual(oa.shape,(96,))
        a.step(action_id([0,0],True));b.step(action_id([0,0],True))
        self.assertFalse(np.array_equal(a.observation(),b.observation()))

    def test_unmeasured_observation_and_phase_change_staleness(self):
        e=Environment(bank(),np.zeros((64,2)))
        e.step(action_id([0,0],True));old=e.last_measurement.copy();oldphase=e.last_phase.copy()
        e.step(action_id([4,0],False));f=e.observation()[-FRAME_DIM:]
        np.testing.assert_array_equal(old,e.last_measurement);np.testing.assert_array_equal(oldphase,e.last_phase)
        self.assertEqual(f[17],1);self.assertEqual(f[18],0);self.assertEqual(f[19],1)
        self.assertAlmostEqual(f[16],1/64)
        e.step(action_id([0,0],True));f=e.observation()[-FRAME_DIM:]
        self.assertEqual(f[16],0);self.assertTrue((f[18:20]==1).all());self.assertEqual(e.last_phase.tolist(),[36,32])

    def test_no_future_measurement_and_same_classical_ppo_cost(self):
        ba=bank();bb=bank();ea=Environment(ba,np.zeros((64,2)));eb=Environment(bb,np.zeros((64,2)))
        c=Classical('periodic');model=ActorCritic()
        with torch.no_grad():model.actor.weight.zero_();model.actor.bias.fill_(-10);model.actor.bias[action_id([0,0],True)]=10
        p=Policy(model);a=c(ea.observation());b=p(eb.observation())
        self.assertEqual(ba.lookups,0);self.assertEqual(bb.lookups,0);self.assertEqual(a,b)
        oa,ra,da,ia=ea.step(a);ob,rb,db,ib=eb.step(b)
        self.assertEqual(ia,ib);self.assertEqual(ba.lookups,4);self.assertEqual(bb.lookups,4)
        self.assertGreater(ia['pilot_tx_energy_pj'],0);self.assertGreater(ia['monitor_energy_pj'],0)
        self.assertGreater(ia['interruption_seconds'],0)

    def test_sampling_support_and_drift_correlation(self):
        with self.assertRaises(ValueError):bank().sample(0,[1.5,.5])
        with self.assertRaises(ValueError):Environment(bank(),np.ones((64,2)))
        d=drift_path(601001,Config());self.assertLessEqual(np.abs(d).max(),.32)
        self.assertGreater(np.corrcoef(d[:-1,0],d[1:,0])[0,1],.8)
        np.testing.assert_array_equal(d,drift_path(601001,Config()))

    def test_block_endpoint_not_best_incumbent(self):
        d=np.zeros((64,2));d[1:]=.3;e=Environment(bank(),d)
        while e.t<64:e.step(action_id([0,0],False))
        r=metrics(e.records);self.assertAlmostEqual(r['violation_rate'],63/64)
        self.assertEqual(r['longest_failure_run'],63);self.assertEqual(r['delivered_bits'],128)
        self.assertEqual(r['calibration_measurements'],0);self.assertEqual(r['tx_energy_pj'],64*32.)

    def test_gae_terminal_has_no_bootstrap_or_cross_episode_return(self):
        a,r=gae(np.array([1.,2.]),np.array([.1,.2]),np.array([999.,999.]),np.ones(2))
        np.testing.assert_allclose(a,[.9,1.8]);np.testing.assert_allclose(r,[1,2])

    def test_small_ppo_updates_and_balances_designs(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(1000);initial=ActorCritic();before={k:v.clone() for k,v in initial.state_dict().items()}
            model=train({'a':bank(),'b':bank()},0,1,Path(tmp)/'train',steps=256)
            self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
            log=json.loads((Path(tmp)/'train/training.json').read_text());self.assertEqual(log['per_design_episodes'],{'a':2,'b':2})
            self.assertEqual(log['blocks'],256);self.assertEqual(log['gradient_steps'],8)
            self.assertEqual(file_hash(Path(tmp)/'train/policy.pt'),log['checkpoint_sha256'])

    def test_operational_surrogate_and_buffer_receive_rollout_metrics(self):
        rows=[dict(u=[.1*i]*6,result=dict(worst_violation_rate=.01*i,worst_downtime_fraction=.1,
            mean_total_energy_pj=2000+100*i,longest_failure_run=i)) for i in range(6)]
        model=DesignSearch(rows);u,pred=model.propose([r['u'] for r in rows]);self.assertEqual(u.shape,(6,));self.assertEqual(pred[0]['id'],0)
        model.observe(dict(u=u.tolist(),result=rows[0]['result']),pred)
        self.assertEqual(len(model.entries),2);self.assertEqual(model.events[0]['gradient_steps'],100)
        self.assertTrue(np.isfinite(model.events[0]['normalized_error_scores']).all())

if __name__=='__main__':unittest.main()
