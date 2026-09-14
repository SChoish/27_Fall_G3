"""Prototype IQL: expectile V, data-action Q, advantage-weighted BC."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coupled_train
from reward_design import RewardSpec


def _offline_cfg():
    return dict(width=8,discount=.95,checkpoint_steps=[4],lr=1e-3,batch_size=8,
                policy_noise=.2,noise_clip=.5,policy_frequency=2,tau=.005,horizon=4,
                margin=.08,primary_ratio=.9)


def _offline_data(n=16):
    rng=np.random.default_rng(7)
    return dict(observations=rng.normal(size=(n,50)).astype('float32'),
                next_observations=rng.normal(size=(n,50)).astype('float32'),
                actions=rng.uniform(-1,1,(n,4)).astype('float32'),
                rewards=rng.normal(size=n).astype('float32'),
                terminals=np.zeros(n,bool),timeouts=np.ones(n,bool))


class IQLTests(unittest.TestCase):
    def test_expectile_weights_positive_and_negative_residuals(self):
        diff=torch.tensor([[2.],[ -2.]])
        loss=coupled_train.expectile_loss(diff,.7)
        self.assertAlmostEqual(float(loss),(.7*4+(1-.7)*4)/2,places=6)
        with self.assertRaisesRegex(ValueError,'expectile'):
            coupled_train.expectile_loss(diff,0.)

    def test_iql_trains_and_policy_loads_without_querying_actor_for_bootstrap(self):
        cfg=_offline_cfg();data=_offline_data();calls={'n':0}
        original=coupled_train.Actor.forward
        def counting(self,x):
            calls['n']+=1;return original(self,x)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);np.savez(out/'dataset.npz',**data);(out/'config.json').write_text(json.dumps(cfg))
            with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
                 patch.object(coupled_train,'verify_artifact'),patch.object(coupled_train,'write_metadata'),\
                 patch.object(coupled_train.Actor,'forward',counting):
                coupled_train.train('iql',0,0.,checkpoint_steps=[4])
            folder=out/'models'/coupled_train.label('iql',0)
            ckpt=torch.load(folder/'step4.pt',weights_only=True)
            self.assertEqual(ckpt['method'],'iql')
            self.assertIn('value',ckpt)
            self.assertEqual(calls['n'],4)
            with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
                 patch.object(coupled_train,'verify_artifact'):
                policy=coupled_train.Policy(folder/'step4.pt')
            action=policy(np.zeros(50,np.float32))
            self.assertEqual(action.shape,(4,))
            self.assertTrue(np.isfinite(action).all())
            meta=json.loads((folder/'training.json').read_text())
            self.assertEqual(meta['actor_optimizer_steps'],4)
            self.assertEqual(meta['value_optimizer_steps'],4)
            self.assertEqual(meta['bootstrap'],'r + gamma * V(next) on data actions only')

    def test_beta_multiplies_advantage_in_actual_training(self):
        cfg=_offline_cfg();data=_offline_data();inputs=[]
        original_exp=torch.exp
        # Initialize lazy optimizer imports before intercepting the exponential.
        torch.optim.Adam(torch.nn.Linear(1,1).parameters())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);np.savez(out/'dataset.npz',**data);(out/'config.json').write_text(json.dumps(cfg))
            for beta in (3.,10.):
                seen=[]
                def record(x):
                    seen.append(x.detach().clone());return original_exp(x)
                with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
                     patch.object(coupled_train,'verify_artifact'),patch.object(coupled_train,'write_metadata'),\
                     patch.object(torch,'exp',side_effect=record):
                    coupled_train.train('iql',0,0.,checkpoint_steps=[1],expectile=.9,temperature=beta)
                self.assertEqual(len(seen),1);inputs.append(seen[0])
            self.assertGreater(float(inputs[0].abs().max()),0.)
            torch.testing.assert_close(inputs[1],inputs[0]*(10./3.))

    def test_iql_accepts_terminal_reward_and_bc_still_rejects_it(self):
        cfg=_offline_cfg();data=_offline_data()
        spec=RewardSpec('terminal_success',4,.95,1.,.08,.9)
        data.update(episode_ids=np.repeat([0,1],8),context_ids=np.repeat([3,5],8),
                    terminals=np.zeros(16,bool),timeouts=np.zeros(16,bool))
        data['timeouts'][7]=True;data['timeouts'][15]=True
        data['next_observations'][:-1]=data['observations'][1:]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);np.savez(out/'dataset.npz',**data);(out/'config.json').write_text(json.dumps(cfg))
            with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
                 patch.object(coupled_train,'verify_artifact'),patch.object(coupled_train,'write_metadata'):
                with self.assertRaisesRegex(ValueError,'BC does not consume reward'):
                    coupled_train.train('bc',0,0.,spec,[2])
                coupled_train.train('iql',1,0.,spec,[2])
            folder=out/'models'/(coupled_train.label('iql',1)+spec.suffix())
            audit=json.loads((folder/'reward_audit.json').read_text())
            self.assertEqual(audit['spec']['mode'],'terminal_success')
            self.assertEqual(audit['selected_transitions'],8)


if __name__=='__main__':unittest.main()
