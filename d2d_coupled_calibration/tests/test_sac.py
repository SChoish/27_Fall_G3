"""Online SAC: empty buffer, terminal success, one update per env step."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coupled_experiment
import coupled_train
from reward_design import RewardSpec


def _cfg(budget=4,steps=8,batch=4):
    ctx=dict(lane0=dict(vdd=1.,rate_gbps=2.),lane1=dict(vdd=1.,rate_gbps=2.))
    return dict(width=8,discount=.95,checkpoint_steps=[steps],lr=1e-3,batch_size=batch,tau=.005,
                horizon=8,margin=.08,primary_ratio=.9,guard_ui=.1,calibration_bits=64,
                contexts=[ctx,ctx],train_ids=[0,1],validation_ids=[2],test_ids=[3])


def _table(path):
    ids=np.array([0,1]);margin=np.full((2,4,4,2,65),.01,np.float32)
    errors=np.ones((2,4,4,2,65),np.int32);energy=np.full((2,4,4),2.,np.float32)
    margin[0,1,1,0,40]=.2;margin[0,1,1,1,30]=.2;errors[0,1,1,0,40]=0;errors[0,1,1,1,30]=0
    energy[0,1,1]=1.
    np.savez(path,ids=ids,margin=margin,errors=errors,energy=energy)


def _run(out,cfg,spec,steps,guard=None):
    def verify(path,*a,**k):
        if guard:guard(path,*a,**k)
    with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
         patch.object(coupled_experiment,'OUT',out),patch.object(coupled_experiment,'config',return_value=cfg),\
         patch.object(coupled_train,'verify_artifact',verify),patch.object(coupled_experiment,'verify_artifact'),\
         patch.object(coupled_train,'write_metadata'):
        coupled_train.train('sac',0,0.,spec,[steps])


class SACTests(unittest.TestCase):
    def test_sac_trains_without_mixed_dataset_and_counts_lookups(self):
        cfg=_cfg();spec=RewardSpec('terminal_success',4,.95,1.,.08,.9)
        def guard(path,*a,**k):
            if Path(path).name=='dataset.npz':raise AssertionError('SAC must not use mixed dataset')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);_table(out/'train_table.npz');(out/'config.json').write_text(json.dumps(cfg))
            self.assertFalse((out/'dataset.npz').exists())
            _run(out,cfg,spec,8,guard)
            folder=out/'models'/(coupled_train.label('sac',0)+spec.suffix())
            meta=json.loads((folder/'training.json').read_text())
            self.assertFalse(meta['offline_dataset'])
            self.assertEqual(meta['environment_steps'],8)
            self.assertEqual(meta['updates_per_env_step'],1)
            self.assertEqual(meta['gradient_updates'],8-cfg['batch_size']+1)
            self.assertEqual(meta['training_episodes'],2)
            self.assertEqual(meta['reference_queries'],2)
            self.assertEqual(meta['policy_queries'],8)
            self.assertEqual(meta['table_lookups'],10)
            self.assertTrue(meta['cached_lookups_counted'])
            self.assertEqual(meta['new_spice_calls'],0)
            self.assertEqual(meta['reward_mode'],'terminal_success')
            audit=json.loads((folder/'reward_audit.json').read_text())
            self.assertEqual(audit['spec']['mode'],'terminal_success')
            with patch.object(coupled_train,'OUT',out),patch.object(coupled_train,'config',return_value=cfg),\
                 patch.object(coupled_train,'verify_artifact'):
                policy=coupled_train.Policy(folder/'step8.pt')
            obs=np.zeros(50,np.float32)
            a=policy(obs);b=policy(obs)
            self.assertEqual(a.shape,(4,));self.assertTrue(np.allclose(a,b))

    def test_potential_or_missing_reward_is_rejected(self):
        cfg=_cfg();bad=RewardSpec('potential_success',4,.95,1.,.08,.9)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);_table(out/'train_table.npz');(out/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'terminal_success'):
                _run(out,cfg,bad,8)
            with self.assertRaisesRegex(ValueError,'terminal_success'):
                _run(out,cfg,None,8)

    def test_one_update_follows_each_env_step_after_warmup(self):
        cfg=_cfg(steps=6,batch=3);spec=RewardSpec('terminal_success',4,.95,1.,.08,.9)
        counts={'n':0};orig=torch.optim.Adam.step
        def counting(self,*a,**k):
            counts['n']+=1;return orig(self,*a,**k)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);_table(out/'train_table.npz');(out/'config.json').write_text(json.dumps(cfg))
            with patch.object(torch.optim.Adam,'step',counting):
                _run(out,cfg,spec,6)
            # actor, critic, alpha each step after warmup
            self.assertEqual(counts['n'],3*(6-3+1))
            meta=json.loads((out/'models'/(coupled_train.label('sac',0)+spec.suffix())/'training.json').read_text())
            self.assertEqual(meta['gradient_updates'],4)


if __name__=='__main__':unittest.main()
