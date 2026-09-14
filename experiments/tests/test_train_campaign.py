"""Regression coverage for simulator discovery and conservative campaign resumption."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_campaign as tc


def executable(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
    return path


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.path_bin=executable(self.root/'system/ngspice')
        self.bundle=executable(self.root/'d2d_coupled_calibration/runtime/usr/bin/ngspice')
        self.env={'PATH':str(self.path_bin.parent)}

    def test_explicit_executable_wins_over_path_and_bundle(self):
        custom=executable(self.root/'custom/spice')
        self.env['NGSPICE_BIN']=str(custom)
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(custom))

    def test_explicit_name_is_resolved_on_path(self):
        self.env['NGSPICE_BIN']='ngspice'
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(self.path_bin))

    def test_relative_explicit_path_survives_child_cwd(self):
        self.env['NGSPICE_BIN']=os.path.relpath(self.path_bin)
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(self.path_bin))

    def test_system_wins_and_works_without_bundle(self):
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(self.path_bin))
        self.bundle.unlink()
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(self.path_bin))

    def test_bundle_is_last_fallback(self):
        self.env['PATH']=''
        self.assertEqual(tc.resolve_ngspice(self.root,self.env),str(self.bundle))

    def test_bad_explicit_path_is_not_silently_replaced(self):
        self.env['NGSPICE_BIN']=str(self.root/'missing')
        with self.assertRaisesRegex(RuntimeError,'NGSPICE_BIN'):tc.resolve_ngspice(self.root,self.env)

    def test_missing_or_nonexecutable_simulator_has_clear_error(self):
        self.env['PATH']='';self.bundle.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError,'install it or set NGSPICE_BIN'):tc.resolve_ngspice(self.root,self.env)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'campaign';self.root.mkdir()
        files={
            'campaign_worker.py':'', 'campaign_report.py':'', 'protocol.json':'{"epochs":500}',
            'adc_driver/configs/stress.json':'{"load_pf":15}',
            'adc_driver/configs/nominal.json':'{}',
            'adc_driver/models/device.spice':'model',
            'd2d_coupled_calibration/coupled_link.py':'',
            'd2d_coupled_calibration/results/coupled_v1/config.json':'{}',
            'adc_driver/job.py':"""import json, os, sys
from pathlib import Path
source,output,flag=map(str,sys.argv[1:4])
counter=Path('calls.txt');counter.write_text(counter.read_text()+'run\\n' if counter.exists() else 'run\\n')
if flag=='fail-first' and len(counter.read_text().splitlines())==1:sys.exit(7)
Path(output).write_text(json.dumps({'input':Path(source).read_text(),'flag':flag,'ngspice':os.environ['NGSPICE_BIN']}))
"""}
        for name,data in files.items():
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
        shutil.copy2(tc.__file__,self.root/'train_campaign.py')
        tc.write_json(self.root/'source_hashes.json',tc.frozen_files(self.root))
        self.binary=executable(Path(self.temp.name)/'system/ngspice')
        self.envpatch=patch.dict(os.environ,{'NGSPICE_BIN':str(self.binary)})
        self.envpatch.start();self.addCleanup(self.envpatch.stop)
        self.data=self.root/'data.json';self.data.write_text('{"value":1}')
        self.output=self.root/'answer.json';self.folder=self.root/'adc_driver'

    def run_job(self,campaign,flag='same',inputs=None):
        campaign.run('job',self.folder,['job.py',self.data,self.output,flag],[self.output],
                     inputs=[self.data] if inputs is None else inputs)

    def test_same_contract_resumes_without_executing_again(self):
        self.run_job(tc.Campaign(self.root));self.run_job(tc.Campaign(self.root))
        self.assertEqual((self.folder/'calls.txt').read_text(),'run\n')
        self.assertEqual(json.loads(self.output.read_text())['ngspice'],str(self.binary))

    def test_changed_command_rejected_even_when_output_hash_matches(self):
        self.run_job(tc.Campaign(self.root));c=tc.Campaign(self.root)
        with self.assertRaisesRegex(ValueError,'command, inputs or outputs changed'):self.run_job(c,'different-seed')
        self.assertEqual((self.folder/'calls.txt').read_text(),'run\n')

    def test_adc_load_and_protocol_are_frozen(self):
        for name,value in [('adc_driver/configs/stress.json','{"load_pf":47}'),('protocol.json','{"epochs":1000}')]:
            with self.subTest(name=name):
                p=self.root/name;old=p.read_text();p.write_text(value)
                with self.assertRaisesRegex(ValueError,'Frozen source/configuration'):tc.Campaign(self.root)
                p.write_text(old)

    def test_missing_config_in_legacy_hash_inventory_is_rejected(self):
        p=self.root/'source_hashes.json';d=json.loads(p.read_text());del d['adc_driver/configs/stress.json'];tc.write_json(p,d)
        with self.assertRaisesRegex(ValueError,'incomplete'):tc.Campaign(self.root)

    def test_changed_dataset_rejected_before_any_new_job(self):
        self.run_job(tc.Campaign(self.root));self.data.write_text('{"value":2}')
        with self.assertRaisesRegex(ValueError,'Job inputs or execution changed'):tc.Campaign(self.root)

    def test_changed_checkpoint_or_added_directory_input_rejected(self):
        models=self.root/'checkpoints';models.mkdir();model=models/'model.pt';model.write_bytes(b'weights1')
        self.run_job(tc.Campaign(self.root),inputs=[self.data,models])
        model.write_bytes(b'weights2')
        with self.assertRaisesRegex(ValueError,'Job inputs or execution changed'):tc.Campaign(self.root)
        model.write_bytes(b'weights1');(models/'extra.pt').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'Job inputs or execution changed'):tc.Campaign(self.root)

    def test_changed_output_is_not_silently_reused_or_overwritten(self):
        self.run_job(tc.Campaign(self.root));self.output.write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'Completed output changed'):tc.Campaign(self.root)
        self.assertEqual(self.output.read_text(),'corrupt')

    def test_changed_simulator_rejected(self):
        self.run_job(tc.Campaign(self.root));self.binary.write_text('#!/bin/sh\nexit 1\n')
        with self.assertRaisesRegex(ValueError,'Execution environment changed'):tc.Campaign(self.root)

    def test_failed_job_can_retry_same_contract(self):
        with self.assertRaisesRegex(RuntimeError,'status 7'):self.run_job(tc.Campaign(self.root),'fail-first')
        self.run_job(tc.Campaign(self.root),'fail-first')
        self.assertEqual(len((self.folder/'calls.txt').read_text().splitlines()),2)

    def test_failed_job_cannot_change_contract(self):
        with self.assertRaises(RuntimeError):self.run_job(tc.Campaign(self.root),'fail-first')
        with self.assertRaisesRegex(ValueError,'command, inputs or outputs changed'):self.run_job(tc.Campaign(self.root),'different-seed')

    def test_input_changed_during_execution_cannot_be_marked_completed(self):
        c=tc.Campaign(self.root)
        code="from pathlib import Path;import sys;Path(sys.argv[1]).write_text('changed');Path(sys.argv[2]).write_text('result')"
        with self.assertRaisesRegex(RuntimeError,'Inputs or execution changed during job'):
            c.run('mutates_input',self.folder,['-c',code,self.data,self.output],[self.output],inputs=[self.data])
        self.assertEqual(c.jobs['mutates_input']['state'],'failed')

    def test_d2d_clone_configuration_must_match_frozen_source(self):
        c=tc.Campaign(self.root);clone=self.root/'d2d_qualities/low'
        shutil.copytree(self.root/'d2d_coupled_calibration',clone)
        c.clone_inputs(clone)
        (clone/'results/coupled_v1/config.json').write_text('{"margin":0.5}')
        with self.assertRaisesRegex(ValueError,'Frozen D2D clone changed'):c.clone_inputs(clone)


if __name__=='__main__':unittest.main()
