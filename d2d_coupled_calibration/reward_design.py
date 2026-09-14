"""Fixed-budget reward ablations using observed calibration features only.

Original Environment and datasets remain unchanged. The finite-horizon reward
uses a common episode prefix; its final transition never bootstraps. Potential
uses the latest observation, so a good incumbent does not erase near misses.
"""
from dataclasses import asdict, dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class RewardSpec:
    mode: str
    budget: int = 16
    gamma: float = .95
    eta: float = 1.
    margin: float = .08
    energy_ratio: float = .9

    def __post_init__(self):
        if self.mode not in ('legacy', 'terminal_success', 'potential_success'):
            raise ValueError('Unknown reward mode')
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 1:
            raise ValueError('budget must be a positive integer')
        if not np.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError('gamma must lie in (0,1]')
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError('eta must be finite and nonnegative')
        if not np.isfinite(self.margin) or self.margin <= 0:
            raise ValueError('margin must be positive')
        if not np.isfinite(self.energy_ratio) or not 0 < self.energy_ratio < 1:
            raise ValueError('energy_ratio must lie in (0,1)')

    def suffix(self):
        return f'_reward_{self.mode}_b{self.budget}_eta{self.eta:g}'


def observations(values):
    x=np.asarray(values,dtype=np.float64)
    if x.shape[-1:]!=(50,) or not np.isfinite(x).all():
        raise ValueError('Expected finite 50-dimensional raw observations')
    return x


def success(obs,spec):
    x=observations(obs)
    return (x[...,40:42].min(-1)>=spec.margin)&(x[...,42:44]==0).all(-1)&(x[...,44]<=spec.energy_ratio)


def potential(obs,spec,terminal=False):
    x=observations(obs)
    # Last of four (action[4], measurement[5]) records: indices 27:36.
    deficit=np.maximum(0, (spec.margin-x[...,31:33])/spec.margin).sum(-1)
    deficit+=x[...,33:35].clip(0,1).mean(-1)
    deficit+=np.maximum(0, x[...,35]/spec.energy_ratio-1)
    phi=-deficit/(1+deficit)  # bounded in [-1,0], no test-label input
    return np.where(terminal,0.,phi)


def incumbent_utility(obs,spec):
    x=observations(obs);m=x[...,40:42];errors=x[...,42:44];energy=x[...,44]
    good=(m.min(-1)>=spec.margin)&(errors==0).all(-1)
    return np.where(good,2-energy,-2-4*np.maximum(0,spec.margin-m).sum(-1)-4*errors.mean(-1)-.1*energy)


def transition_reward(obs,next_obs,terminal,spec):
    if spec.mode=='legacy':
        return incumbent_utility(next_obs,spec)-incumbent_utility(obs,spec)-.01
    base=np.asarray(terminal,dtype=np.float64)*success(next_obs,spec)
    if spec.mode=='terminal_success':return base
    return base+spec.eta*(spec.gamma*potential(next_obs,spec,terminal)-potential(obs,spec))


def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def relabel(data,spec):
    """Select identical first-B prefixes for every mode, preserving observations.

    Original remaining-time feature (32-t)/32 is retained in every mode, including at rollout;
    it still uniquely encodes the clock of the fixed B-step task. Original
    training packets and the incumbent rule are untouched. No SPICE is called.
    """
    n=len(data['rewards']);ids=np.asarray(data['episode_ids'])
    if ids.shape!=(n,) or n==0:raise ValueError('Invalid episode IDs')
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1];ends=np.r_[starts[1:],n]
    if len(np.unique(ids[starts]))!=len(starts):raise ValueError('Episodes must be contiguous')
    selected=[]
    for start,end in zip(starts,ends):
        if end-start<spec.budget:raise ValueError('Episode shorter than requested budget')
        stop=start+spec.budget
        if np.any(data['terminals'][start:stop-1]) or np.any(data['timeouts'][start:stop-1]):
            raise ValueError('Episode ends before requested budget')
        if not np.array_equal(data['next_observations'][start:stop-1],data['observations'][start+1:stop]):
            raise ValueError('Broken episode continuity')
        if np.unique(data['context_ids'][start:stop]).size!=1:raise ValueError('Context changed within episode')
        selected.extend(range(start,stop))
    ix=np.asarray(selected);result={k:np.array(v[ix],copy=True) for k,v in data.items()}
    terminal=np.zeros(len(ix),dtype=bool);terminal[spec.budget-1::spec.budget]=True
    # This is the end of the defined task, not a truncation of an infinite task.
    result['terminals']=terminal;result['timeouts']=np.zeros(len(ix),dtype=bool)
    computed=transition_reward(result['observations'],result['next_observations'],terminal,spec)
    if spec.mode=='legacy':
        if not np.allclose(computed,result['rewards'],atol=4e-6,rtol=0):
            raise ValueError('Source reward does not match the legacy incumbent objective')
        # Preserve original measured rewards, avoiding float32 reconstruction drift.
    else:result['rewards']=np.asarray(computed,dtype=np.float32)
    discounted=np.asarray(result['rewards'],float).reshape(-1,spec.budget)@(spec.gamma**np.arange(spec.budget))
    endpoint_success=success(result['next_observations'][terminal],spec)
    identity_error=None
    if spec.mode!='legacy':
        expected=spec.gamma**(spec.budget-1)*endpoint_success.astype(float)
        if spec.mode=='potential_success':expected-=spec.eta*potential(result['observations'][::spec.budget],spec)
        identity_error=float(np.max(np.abs(discounted-expected)))
        if identity_error>2e-6:raise ValueError('Discounted shaping identity failed')
    audit=dict(spec=asdict(spec),source_transitions=n,selected_transitions=len(ix),episodes=len(starts),
        terminal_success_episodes=int(endpoint_success.sum()),identity_max_abs_error=identity_error,
        reward_min=float(result['rewards'].min()),reward_max=float(result['rewards'].max()),
        reward_mean=float(result['rewards'].mean()),reward_std=float(result['rewards'].std()),
        nonterminal_reward_std=float(result['rewards'][~terminal].std()) if spec.budget>1 else 0.,
        nonreward_array_hashes={k:array_hash(v) for k,v in result.items() if k!='rewards'},
        clock_encoding='Original observation clock retained identically in all modes; stop after budget steps',
        supervision='Training calibration observations only; no independent packets or test contexts')
    return result,audit
