"""A preserved baseline must share the actual task and training arrays."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('iql_compare',Path(__file__).parents[1]/'train_iql_compare.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class IQLCompareTests(unittest.TestCase):
    def fixture(self):
        p=dict(quality='mixed',source_hashes={'dataset':'a'},seeds=[0,1,2],budget=16,eta=1.,discount=.95,
               primary_step=50000,steps=[5000,15000,50000],method='iql',reward_mode='terminal_success',expectile=.9,temperature=10.)
        old=copy.deepcopy(p);old.update(expectile=.7,temperature=3.)
        arrays={'observations':'one','actions':'two'}
        return p,dict(protocol=old,reward_audits={'s0':{'nonreward_array_hashes':arrays}}),arrays
    def test_different_pair_is_allowed_for_same_task(self):
        p,ref,arrays=self.fixture();runner.validate_reference(ref,p,arrays,iql=True)
    def test_budget_or_data_change_cannot_be_presented_as_pair_only(self):
        for key,value in [('budget',32),('source_hashes',{'dataset':'changed'})]:
            p,ref,arrays=self.fixture();ref['protocol'][key]=value
            with self.assertRaisesRegex(ValueError,key):runner.validate_reference(ref,p,arrays,iql=True)
    def test_same_source_but_different_relabelled_arrays_is_rejected(self):
        p,ref,arrays=self.fixture();ref['reward_audits']['s0']['nonreward_array_hashes']={'observations':'changed'}
        with self.assertRaisesRegex(ValueError,'arrays'):runner.validate_reference(ref,p,arrays,iql=True)

if __name__=='__main__':unittest.main()
