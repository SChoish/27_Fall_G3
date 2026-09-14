"""SAC campaign copies lookup tables only and does not treat 50k as equal cost."""
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('sac_online',Path(__file__).parents[1]/'train_sac_online.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class SACOnlineTests(unittest.TestCase):
    def test_initialize_omits_mixed_dataset_and_keeps_cost_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            cfg=dict(discount=.95,checkpoint_steps=[5000,15000,50000],train_ids=[0],validation_ids=[1],test_ids=[2])
            (source/'config.json').write_text(json.dumps(cfg))
            for split in ('train','validation'):
                np.savez(source/f'{split}_table.npz',ids=np.array([0 if split=='train' else 1]))
                (source/f'{split}_table.npz.metadata.json').write_text('{}\n')
            (source/'dataset.npz').write_bytes(b'offline-should-not-copy')
            root=Path(tmp)/'camp';runner.initialize(source,root)
            out=root/'d2d/results/coupled_v1'
            self.assertFalse((out/'dataset.npz').exists())
            self.assertTrue((out/'train_table.npz').exists())
            protocol=json.loads((root/'protocol.json').read_text())
            self.assertFalse(protocol['offline_dataset'])
            self.assertIn('not the same cost',protocol['cost_note'])
            self.assertNotIn('dataset.npz',protocol['source_hashes'])
            self.assertEqual(runner.SOURCE_NAMES[0],'config.json')
            self.assertTrue(all('dataset' not in name for name in runner.SOURCE_NAMES))

    def test_shared_table_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp); (source/'config.json').write_text('a')
            (source/'train_table.npz').write_bytes(b'one');(source/'validation_table.npz').write_bytes(b'two')
            ref=dict(protocol=dict(source_hashes={'config.json':runner.sha(source/'config.json'),
                'train_table.npz':'0'*64,'validation_table.npz':runner.sha(source/'validation_table.npz')}))
            with self.assertRaisesRegex(ValueError,'train_table'):
                runner.shared_tables(source,ref)


if __name__=='__main__':unittest.main()
