"""Reward semantics, finite-horizon telescoping and identical-data controls."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reward_design import RewardSpec,potential,success,transition_reward,relabel


def obs(margin=.1,errors=0.,energy=1.,best_margin=.1,best_energy=1.):
    x=np.zeros(50);x[31:36]=[margin,margin,errors,errors,energy]
    x[40:45]=[best_margin,best_margin,0,0,best_energy];x[-1]=1.
    return x


def dataset():
    states=[obs(),obs(-.3,.4,.8),obs(.079,0,.8),obs(.1,0,.85,best_energy=.85),obs(.1,0,.8,best_energy=.8)]
    for i,x in enumerate(states):x[-1]=1-i/4
    before=np.array(states[:-1],np.float32);after=np.array(states[1:],np.float32)
    rewards=transition_reward(before,after,np.array([False]*3+[True]),RewardSpec('legacy')).astype(np.float32)
    return dict(observations=np.tile(before,(2,1)),next_observations=np.tile(after,(2,1)),
        actions=np.zeros((8,4),np.float32),rewards=np.tile(rewards,2),terminals=np.zeros(8,bool),
        timeouts=np.tile([False,False,False,True],2),episode_ids=np.repeat([2,9],4),context_ids=np.repeat([1,4],4))


class RewardTests(unittest.TestCase):
    def test_legacy_loses_failure_severity_but_potential_preserves_it(self):
        before=obs();severe=obs(-.2,.5,.9);near=obs(.079,0,.9)
        old=RewardSpec('legacy');new=RewardSpec('potential_success')
        self.assertAlmostEqual(float(transition_reward(before,severe,False,old)),-.01)
        self.assertAlmostEqual(float(transition_reward(before,near,False,old)),-.01)
        self.assertLess(float(transition_reward(before,severe,False,new)),float(transition_reward(before,near,False,new)))

    def test_terminal_uses_returned_incumbent_not_latest_measurement(self):
        s=RewardSpec('terminal_success');before=obs()
        latest_bad_but_incumbent_good=obs(-.1,.5,1.,best_energy=.85)
        latest_good_but_incumbent_bad=obs(.1,0,.85,best_margin=.07,best_energy=.85)
        self.assertEqual(float(transition_reward(before,latest_bad_but_incumbent_good,True,s)),1.)
        self.assertEqual(float(transition_reward(before,latest_good_but_incumbent_bad,True,s)),0.)
        self.assertEqual(float(transition_reward(before,latest_bad_but_incumbent_good,False,s)),0.)

    def test_terminal_potential_is_zero_even_for_failed_endpoint(self):
        s=RewardSpec('potential_success');bad=obs(-.5,.5,1.2)
        self.assertLess(float(potential(bad,s)),0.)
        self.assertEqual(float(potential(bad,s,terminal=True)),0.)

    def test_discounted_telescoping_for_multiple_budgets_and_discounts(self):
        rng=np.random.default_rng(712)
        for gamma in [1.,.95,.5]:
            for budget in [1,3,16,32]:
                s=RewardSpec('potential_success',budget,gamma,eta=.7)
                history=[obs()]+[obs(float(rng.uniform(-.3,.2)),float(rng.uniform(0,1)),float(rng.uniform(.7,1.2)),best_energy=.85) for _ in range(budget)]
                rewards=[float(transition_reward(history[t],history[t+1],t==budget-1,s)) for t in range(budget)]
                expected=gamma**(budget-1)*float(success(history[-1],s))-s.eta*float(potential(history[0],s))
                self.assertAlmostEqual(sum(gamma**t*r for t,r in enumerate(rewards)),expected,places=12)

    def test_loops_do_not_improve_discounted_episode_return(self):
        s=RewardSpec('potential_success',3);start=obs();bad=obs(-.3,.4,.8);near=obs(.079,0,.8);end=obs(best_energy=.85)
        def total(path):return sum(s.gamma**t*float(transition_reward(path[t],path[t+1],t==2,s)) for t in range(3))
        self.assertAlmostEqual(total([start,bad,near,end]),total([start,near,bad,end]),places=12)

    def test_terminal_objective_prefers_success_probability_over_mean_energy(self):
        s=RewardSpec('terminal_success');start=obs()
        a=float(transition_reward(start,obs(best_energy=.89),True,s))
        b=np.mean([transition_reward(start,obs(best_energy=e),True,s) for e in [.7,1.]])
        self.assertGreater(a,b)

    def test_relabel_changes_only_rewards_for_matched_prefixes(self):
        source=dataset();copies={k:v.copy() for k,v in source.items()};results=[]
        for mode in ['legacy','terminal_success','potential_success']:
            data,audit=relabel(source,RewardSpec(mode,3));results.append((data,audit))
            self.assertEqual(audit['selected_transitions'],6)
            np.testing.assert_array_equal(data['terminals'],[False,False,True,False,False,True])
            self.assertFalse(data['timeouts'].any())
            mask=~(data['terminals']|data['timeouts']);self.assertFalse(mask[2]);self.assertTrue(mask[1])
        self.assertEqual(results[0][1]['nonreward_array_hashes'],results[1][1]['nonreward_array_hashes'])
        self.assertEqual(results[1][1]['nonreward_array_hashes'],results[2][1]['nonreward_array_hashes'])
        for k in copies:np.testing.assert_array_equal(source[k],copies[k])
        self.assertLess(results[2][1]['identity_max_abs_error'],2e-6)

    def test_relabel_rejects_short_discontinuous_and_interleaved_episodes(self):
        with self.assertRaisesRegex(ValueError,'shorter'):relabel(dataset(),RewardSpec('terminal_success',5))
        d=dataset();d['next_observations'][0,0]=1
        with self.assertRaisesRegex(ValueError,'continuity'):relabel(d,RewardSpec('terminal_success',3))
        d=dataset();d['episode_ids'][4]=2;d['episode_ids'][6]=2
        with self.assertRaisesRegex(ValueError,'contiguous'):relabel(d,RewardSpec('terminal_success',1))

    def test_eta_zero_matches_terminal_only(self):
        s=RewardSpec('potential_success',eta=0.);before=obs();after=obs(best_energy=.85)
        self.assertEqual(float(transition_reward(before,after,True,s)),float(transition_reward(before,after,True,replace(s,mode='terminal_success'))))

    def test_boundary_success_and_potential_bounds(self):
        s=RewardSpec('potential_success')
        self.assertTrue(success(obs(best_margin=.08,best_energy=.9),s))
        self.assertFalse(success(obs(best_margin=.0799,best_energy=.9),s))
        for x in [obs(),obs(-1e8,1,1e8),obs(.1,0,.7)]:self.assertTrue(-1<=potential(x,s)<=0)

    def test_invalid_specs(self):
        for kwargs in [{'mode':'other'},{'budget':0},{'gamma':0},{'gamma':float('nan')},{'eta':-1},{'margin':0},{'energy_ratio':1}]:
            with self.assertRaises(ValueError):RewardSpec(**dict({'mode':'terminal_success'},**kwargs))


if __name__=='__main__':unittest.main()
