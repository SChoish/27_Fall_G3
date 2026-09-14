"""Contract, non-default margin and simulator failure regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coupled_experiment as experiment
import coupled_link as coupled
import link
import provenance


class ProvenanceTests(unittest.TestCase):
    def test_original_tables_still_load(self):
        for split in ('train', 'validation', 'test'):
            table = experiment.Table(split)
            self.assertEqual(table.ids, table.cfg[split + '_ids'])

    def test_table_rejects_changed_measurement_or_circuit(self):
        for field in ('guard', 'coupling'):
            cfg = copy.deepcopy(experiment.config())
            if field == 'guard':
                cfg['guard_ui'] = .2
            else:
                cfg['contexts'][0]['coupling_pf'] += .1
            with patch.object(experiment, 'config', return_value=cfg):
                with self.assertRaisesRegex(ValueError, 'contract'):
                    experiment.Table('train')

    def test_metadata_binds_content_configuration_code_and_dependencies(self):
        cfg = experiment.config()
        with tempfile.TemporaryDirectory(dir=provenance.ROOT) as tmp:
            artifact = Path(tmp) / 'model.pt'
            dependency = Path(tmp) / 'dataset.npz'
            artifact.write_bytes(b'model'); dependency.write_bytes(b'dataset')
            provenance.write_metadata(artifact, cfg, dependencies=[dependency])
            provenance.verify_artifact(artifact, cfg, dependencies=[dependency])
            changed = dict(cfg, guard_ui=.2)
            with self.assertRaisesRegex(ValueError, 'contract'):
                provenance.verify_artifact(artifact, changed)
            current = provenance.contract(cfg)
            current['model_sha256'] = 'different model'
            with patch.object(provenance, 'contract', return_value=current):
                with self.assertRaisesRegex(ValueError, 'contract'):
                    provenance.verify_artifact(artifact, cfg)
            dependency.write_bytes(b'changed dataset')
            with self.assertRaisesRegex(ValueError, 'dependency mismatch'):
                provenance.verify_artifact(artifact, cfg)
            dependency.write_bytes(b'dataset'); artifact.write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                provenance.verify_artifact(artifact, cfg)

    def test_new_table_dataset_checkpoint_roundtrip(self):
        import coupled_train
        cfg = dict(experiment.config(), train_ids=[0], horizon=2,
                   episodes_per_context=2, checkpoint_steps=[2], width=8, batch_size=2)
        with tempfile.TemporaryDirectory(dir=provenance.ROOT) as tmp:
            folder = Path(tmp)
            cfg_path = folder / 'config.json'
            cfg_path.write_text(json.dumps(cfg))
            with patch.object(experiment, 'OUT', folder), patch.object(
                    experiment, 'config', return_value=cfg), patch.object(
                    coupled_train, 'OUT', folder), patch.object(coupled_train, 'config', return_value=cfg):
                wave = {'energy_pj_bit': 1., 'seconds': 0.}
                with patch.object(experiment, 'cached_wave', return_value=(None, None, wave)), patch.object(
                        experiment, 'phase_table', return_value=(np.full((2, 65), .12), np.zeros((2, 65), dtype=np.int16))):
                    experiment.prepare('train', workers=1)
                self.assertEqual(experiment.Table('train').ids, [0])
                experiment.collect()
                coupled_train.train('bc', 0, 0.)
                checkpoint = folder / 'models/bc_s0/step2.pt'
                policy = coupled_train.Policy(checkpoint)
                action = policy(np.zeros(50, dtype=np.float32))
                self.assertEqual(action.shape, (4,))
                self.assertTrue(np.isfinite(action).all())
                with (folder / 'dataset.npz').open('ab') as f:
                    f.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'dependency mismatch'):
                    coupled_train.Policy(checkpoint)

    def test_unknown_legacy_artifact_is_never_blessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'unknown.npz'; p.write_bytes(b'unknown')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                provenance.verify_artifact(p, experiment.config())
            self.assertFalse(provenance.metadata_path(p).exists())

    def test_policy_rejects_configuration_before_loading_weights(self):
        import coupled_train
        cfg = dict(experiment.config(), guard_ui=.2)
        path = experiment.OUT / 'models/bc_s0/step4000.pt'
        with patch.object(coupled_train, 'config', return_value=cfg), patch.object(
                coupled_train.torch, 'load', side_effect=AssertionError('Weights must not load')):
            with self.assertRaisesRegex(ValueError, 'contract'):
                coupled_train.Policy(path)


class MarginTests(unittest.TestCase):
    @staticmethod
    def measurement(margin, energy):
        return dict(margins=[margin, margin], errors=[0, 0], energy_ratio=energy, bits_per_lane=64)

    def test_coordinate_retains_improvement_under_configured_margin(self):
        cfg = dict(experiment.config(), margin=.1)
        with patch.object(experiment, 'config', return_value=cfg):
            search = experiment.Search('traditional')
        initial = coupled.CoupledSetting(1, 32, 1, 32)
        search.observe(initial, self.measurement(.09, .8))
        first = search.suggest(); search.observe(first, self.measurement(.12, 1.))
        second = search.suggest(); search.observe(second, self.measurement(.01, .7))
        third = search.suggest()
        self.assertEqual(search.best(), first)
        self.assertEqual(third.strength0, first.strength0)

    def test_ga_and_bo_use_configured_margin_at_every_score(self):
        cfg = dict(experiment.config(), margin=.1)
        for method, function in [('ga', 'rank'), ('bo', 'utility')]:
            with patch.object(experiment, 'config', return_value=cfg):
                search = experiment.Search(method, params={'population': 4})
            for phase in range(10, 60, 10):
                search.observe(coupled.CoupledSetting(3, phase, 3, 32), self.measurement(.12, .9))
            original = getattr(experiment, function)
            def checked(measurement, margin=None):
                self.assertEqual(margin, .1)
                return original(measurement, margin)
            with patch.object(experiment, function, side_effect=checked):
                search.suggest()

    def test_validation_tiebreak_uses_configured_margin(self):
        import coupled_evaluate
        cfg = dict(experiment.config(), margin=.1)
        rows = [dict(calibration_quality=False, calibration=self.measurement(.09, .8), budget=32)]
        with patch.object(coupled_evaluate, 'config', return_value=cfg):
            score = coupled_evaluate.score(rows)
        self.assertLess(score[1], 0.)


class SimulatorFailureTests(unittest.TestCase):
    def check_failure(self, module, write_new):
        bits = coupled.payloads(42, 8)
        context = coupled.CoupledContext(link.Context(), link.Context())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'wave.txt').write_text('old waveform')
            (p / 'wave.npz').write_bytes(b'old compressed waveform')
            def failed_run(*args, **kwargs):
                self.assertFalse((p / 'wave.txt').exists())
                self.assertFalse((p / 'wave.npz').exists())
                if write_new:
                    (p / 'wave.txt').write_text('partial new output')
                return SimpleNamespace(returncode=0, stdout='Error: transient failed', stderr='')
            with patch.object(link, 'simulator', return_value=('unused', {})), patch.object(
                    module.subprocess, 'run', side_effect=failed_run):
                with self.assertRaisesRegex(RuntimeError, 'transient failed'):
                    if module is coupled:
                        module.simulate(context, (8, 8), bits, keep=p)
                    else:
                        module.simulate(context.lane0, 8, bits[0], keep=p)
            self.assertIn('transient failed', (p / 'ngspice.log').read_text())
            self.assertFalse((p / 'wave.npz').exists())

    def test_no_reuse_of_old_output(self):
        for module in (link, coupled):
            self.check_failure(module, False)

    def test_error_log_rejected_even_with_new_output_and_zero_exit(self):
        for module in (link, coupled):
            self.check_failure(module, True)


if __name__ == '__main__':
    unittest.main()
