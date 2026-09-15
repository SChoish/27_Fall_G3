"""Software-only end-to-end plumbing test; no transistor performance evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from d2d_joint import run
from d2d_joint.physics import PhaseBank
from d2d_sizing.circuit import file_hash,write_json


class PipelineTest(unittest.TestCase):
    def test_four_comparisons_use_locked_designs_and_policy_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'synthetic_run'
            run.initialize(root,'/usr/bin/true')
            p=run.read(root/'protocol.json');p['ppo_steps_per_stage']=768
            p['development_drift_seeds']=[601001];p['test_drift_seeds']=[901001]
            write_json(root/'protocol.json',p)
            hashes=run.read(root/'input_hashes.json');hashes['protocol.json']=file_hash(root/'protocol.json');write_json(root/'input_hashes.json',hashes)
            physical=[]
            def synthetic(x,cases,output,workers):
                Path(output).mkdir(parents=True);physical.append((list(x),[c.name for c in cases]))
                phase=np.linspace(-.5,1.5,513);bits=np.tile(np.arange(64)%2,(4,2,1))
                voltage=np.broadcast_to(.5+(2*bits[:,:,:,None]-1)*.3,(4,2,64,513)).copy()
                energy=float(np.sum(x)/100)
                bank=PhaseBank(phase,voltage,bits,[energy]*4,[0.]*4,cases,x)
                result=dict(valid=True,feasible=True,margins=[.3,.3],energy_pj_bit=energy,overshoot_fraction=0.)
                return bank,result,dict(simulator_version=p['simulator_version'],spice_transient_attempts=0,cache_hits=0)
            with patch.object(run,'characterize',synthetic),patch.object(run,'__file__',str(root/'code/d2d_joint/run.py')),contextlib.redirect_stdout(io.StringIO()):
                run.execute(root)
            self.assertEqual(run.read(root/'status.json')['state'],'completed')
            search=run.read(root/'design_search.json');self.assertEqual(len(search['designs']),12)
            self.assertTrue(all(len(pool)==8 for pool in search['pools'].values()))
            locked=run.read(root/'locked_selection.json');self.assertEqual(len(locked),8)
            self.assertEqual(set(locked),set(run.read(root/'final_results.json')))
            for row in locked.values():
                self.assertEqual(row['x'],search['designs'][row['design']])
                seed=row['policy_seed']
                if seed is not None:
                    self.assertEqual(row['policy_sha256'],file_hash(root/'policies'/f'v2_s{seed}'/'policy.pt'))
                    train=run.read(root/'policies'/f'v2_s{seed}'/'training.json')
                    self.assertEqual(set(train['per_design_episodes'].values()),{1})
            costs=run.read(root/'costs.json');self.assertEqual(len(costs['training']),6)
            self.assertEqual(sum(x['blocks'] for x in costs['training']),4608)
            self.assertTrue(all(x['spice_transient_attempts']==0 for x in costs['physical']))

if __name__=='__main__':unittest.main()
