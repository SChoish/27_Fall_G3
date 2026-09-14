import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from circuit import Context, NAMES, aggregate
from critics import CriticBuffer, CriticOptions, selection_probabilities
from learning import Model, LocalSearch
from attention import SearchOptions
from search import fingerprint, optimize
from round2 import relabel


def oracle(error=1.2,energy=100.,valid=True):
    return aggregate([dict(context=asdict(Context()),valid=valid,feasible=valid and error<=1.,
                           worst_hold_error_mv=error,mean_driver_energy_pj=energy,
                           normalized_violation=max(0.,error-1.))])


class CriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.source=self.folder/'source.json'
        rng=np.random.default_rng(1)
        self.rows=[dict(index=i,query=i+1,u=rng.uniform(0,1,len(NAMES)).tolist(),
                        split=0 if i<6 else 1 if i<8 else 2,oracle=oracle(.5+i*.2,100+i),cost=1+i*.1) for i in range(10)]
        data=dict(records=self.rows,context_hash=fingerprint([Context()]))
        self.source.write_text(json.dumps(data))
        self.payload=dict(state_dict=Model().state_dict(),mean=torch.tensor([0.,2.]),std=torch.tensor([.5,.2]),
                          method='rnc',seed=0,source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                          contexts=[asdict(Context())],context_hash=data['context_hash'],source_design_queries=10)
        self.ckpt=self.folder/'model.pt';torch.save(self.payload,self.ckpt)

    def options(self,mode='buffer',**kw):
        return CriticOptions(mode=mode,source_data=str(self.source),epochs=1,**kw)

    def test_selection_sign_prefers_lower_error(self):
        p=selection_probabilities([.1,1.,10.],.25)
        self.assertTrue(p[0]>p[1]>p[2]);self.assertAlmostEqual(p.sum(),1.)

    def test_full_update_replay_excludes_validation_and_test(self):
        buf=CriticBuffer(self.payload,self.options())
        self.assertEqual([r['index'] for r in buf.offline],list(range(6)))
        before=copy.deepcopy(buf.current['state_dict'])
        event=buf.update(self.rows[:4])
        after=buf.current['state_dict']
        for prefix in ('encoder','regressor','classifier','projector'):
            self.assertTrue(any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith(prefix)),prefix)
        torch.testing.assert_close(buf.current['mean'],self.payload['mean'])
        self.assertEqual(event['online_rows'],4)
        self.assertEqual(buf.current['evidence'],[])
        self.assertFalse(buf.update(self.rows[:4])['updated'])

    def test_evidence_is_version_specific_and_preserves_predictions(self):
        buf=CriticBuffer(self.payload,self.options(capacity=2))
        pred=buf.predict(self.rows[0]['u']);saved=copy.deepcopy(pred)
        buf.observe(self.rows[0],pred,None)
        self.assertEqual(pred,saved)
        self.assertEqual(len(buf.current['evidence']),1)
        buf.update(self.rows[:4]);self.assertEqual(buf.current['evidence'],[])
        parent=next(e for e in buf.entries if e['id']==0)
        self.assertEqual(len(parent['evidence']),1)
        for n in (5,6,7):
            row=self.rows[n-1];buf.observe(row,buf.predict(row['u']),1.)
            buf.update(self.rows[:n])
            self.assertLessEqual(len(buf.entries),2)
        path=self.folder/'buffer.pt';buf.save(path)
        saved=torch.load(path,weights_only=False)
        self.assertEqual(saved['current_id'],buf.current['id'])
        self.assertIn('projector.2.weight',saved['entries'][-1]['state_dict'])

    def test_frozen_and_latest_modes(self):
        frozen=CriticBuffer(self.payload)
        self.assertFalse(frozen.update(self.rows)['updated'])
        for k,v in self.payload['state_dict'].items():torch.testing.assert_close(v,frozen.current['state_dict'][k])
        latest=CriticBuffer(self.payload,self.options('latest'))
        latest.update(self.rows[:4]);first=latest.current['id'];event=latest.update(self.rows[:5])
        self.assertEqual(len(latest.entries),1);self.assertEqual(event['parent_id'],first)

    def test_source_mismatch_rejected(self):
        self.source.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA256'):CriticBuffer(self.payload,self.options())

    def test_margin_reuses_physics_but_new_inputs_are_rejected(self):
        r=oracle(.9)
        updated=relabel(r,[replace(Context(),error_limit_mv=.8)])
        self.assertTrue(r['feasible']);self.assertFalse(updated['feasible'])
        self.assertAlmostEqual(updated['normalized_violation'],.125)
        local=LocalSearch(self.ckpt,[replace(Context(),error_limit_mv=.8)])
        self.assertAlmostEqual(local.limit,np.log10(.8))
        with self.assertRaises(ValueError):relabel(r,[replace(Context(),levels_v=(.9,1.))])
        with self.assertRaises(ValueError):LocalSearch(self.ckpt,[replace(Context(),acquisition_ns=100.)])

    def test_prediction_written_before_spice_and_failed_calls_are_charged(self):
        output=self.folder/'search';calls=[]
        def fake(x,contexts):
            pending=json.loads((output/'pending_prediction.json').read_text())
            self.assertEqual(pending['query'],len(calls)+1)
            self.assertTrue(pending['predictions'])
            calls.append(x)
            return oracle(valid=len(calls)%3!=0)
        with patch('search.evaluate_suite',side_effect=fake),patch('search.verify_runtime'):
            r=optimize('rnc',16,0,[Context()],output,self.ckpt,
                       SearchOptions(attention='off',search_steps=1),self.options())
        self.assertEqual(len(calls),16);self.assertEqual(r['transient_queries'],16)
        self.assertEqual(len(r['trace'][8]['critic_predictions']),2)
        self.assertEqual(r['trace'][8]['proposal']['critic']['after_query'],8)
        self.assertFalse((output/'pending_prediction.json').exists())


if __name__=='__main__':unittest.main()

class FinalReportTests(unittest.TestCase):
    def test_test_suite_is_locked_before_measurement_and_energy_is_conditional(self):
        import round2
        from circuit import write_json
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);index=[]
            for seed in (0,1):
                trace=out/f'trace{seed}.json';dev=out/f'dev{seed}.json'
                write_json(trace,dict(best=dict(x=[1.]*9,query=9),transient_queries=64,
                                      first_success=4,contexts=[asdict(Context())]*4))
                write_json(dev,dict(transient_queries=0))
                index.append(dict(name=f'C_s{seed}',condition='C',method='rnc',mode='buffer',attention='learned',
                                  seed=seed,trace=str(trace),development=str(dev),trace_sha256=round2.sha(trace),
                                  development_sha256=round2.sha(dev)))
            source=out/'source.json';write_json(source,dict(transient_queries=512))
            write_json(out/'runs.json',index);write_json(out/'attention_complete.json',{})
            write_json(out/'data_C.json',dict(reuse=dict(additional_transient_queries=819)))
            write_json(out/'diagnostic.json',dict(transient_queries=134))
            p=dict(source=str(source),final_contexts=[asdict(Context()),asdict(replace(Context(),levels_v=(.9,1.)))])
            calls=[]
            def fake(x,cs):
                self.assertEqual(len(round2.read(out/'endpoints.json')),2)
                self.assertTrue(all(c.error_limit_mv==1. for c in cs))
                calls.append(x);return oracle(.9 if len(calls)==1 else 1.1,120. if len(calls)==1 else 80.)
            with patch('round2.evaluate_suite',side_effect=fake):round2.final_test(out,p)
            result=round2.read(out/'summary.json');group=result['groups']['C_rnc_buffer_learned']
            self.assertEqual(group['all_inputs_pass_rate'],.5)
            self.assertEqual(group['passing_energy_mean_pj'],120.)
            self.assertEqual(group['runs'][0]['returned_design_found_at_transients'],36)
            self.assertIsNone(group['runs'][1]['energy_pj'])
            self.assertEqual(result['costs']['final_test_transients'],4)
            self.assertEqual(result['costs']['original_shared_offline_transients'],512)
            # Resuming a complete final test cannot spend more measurements.
            with patch('round2.evaluate_suite',side_effect=AssertionError('reran completed test')):round2.final_test(out,p)

    def test_attempt_record_precedes_simulation(self):
        import round2
        from circuit import write_json
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def fake(x,c,output,timeout):
                paths=list((out/'spice_attempts').glob('*.json'))
                self.assertEqual(len(paths),1)
                self.assertFalse(round2.read(paths[0])['completed'])
                return dict(valid=False,elapsed_seconds=.1)
            original=round2.circuit.evaluate
            try:
                with patch('round2.evaluate',side_effect=fake):charged=round2.install_accounting(out)
                charged([1.]*9)
                record=round2.read(next((out/'spice_attempts').glob('*.json')))
                self.assertTrue(record['completed']);self.assertFalse(record['valid'])
            finally:round2.circuit.evaluate=original
