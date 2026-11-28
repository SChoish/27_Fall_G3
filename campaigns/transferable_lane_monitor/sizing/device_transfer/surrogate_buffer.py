"""Select target-calibrated SPICE surrogate heads using prequential electrical measurements."""
import numpy as np

from blind_width_policy import kernel, rank_candidates

RULES = ("mae", "near_goal", "ranking", "false_pass")
ARMS = ("mlp200", "fresh", "latest", "random_head", *RULES)
WINDOW = 4
RIDGE = .01


def margins(y, goal):
    out = y.copy()
    out[:, 0] -= np.log(goal)
    return out


def merit(y, goal):
    # Negative iff all five electrical specifications are satisfied.
    return margins(y, goal).max(axis=1)


def losses(rule, predicted, measured, goal, scale):
    errors = np.abs(predicted - measured) / scale
    mae = errors.mean(axis=1)
    actual = merit(measured, goal)
    estimated = merit(predicted, goal)
    if rule == "mae":
        return (mae.mean(),)
    if rule == "near_goal":
        weights = 1 / (.03 + np.abs(actual))
        return (np.average(mae, weights=weights),)
    if rule == "ranking":
        i, j = np.triu_indices(len(actual), k=1)
        signs = np.sign(actual[i]-actual[j])
        eligible = signs != 0
        agreement = signs * np.sign(estimated[i]-estimated[j])
        return (np.sum((1-agreement[eligible])/2), mae.mean())
    if rule == "false_pass":
        # Count per-metric violations to give feedback even when no design passes all five.
        ok = margins(measured, goal) <= 0
        pred_ok = margins(predicted, goal) <= 0
        return (3*np.sum(pred_ok & ~ok) + np.sum(~pred_ok & ok), mae.mean())
    raise ValueError(rule)


def fit_head(feature, residual, end):
    x = feature[:end]
    return np.linalg.solve(x.T @ x + RIDGE*np.eye(x.shape[1]), x.T @ residual[:end])


def select_head(arm, net, z, y, lo, hi, goal, seed):
    feature = net.encode(z)
    base = y[0] + net.readout(feature)
    residual = y - base
    stop = len(y) - WINDOW
    # Reconstructions of immutable checkpoints: training includes rows [0,end).
    # The newest eligible and the three older heads have never seen the shared window.
    ends = list(range(stop, max(0, stop-8), -2))
    if arm == "fresh":
        end = len(y)
        return fit_head(feature, residual, end), end
    heads = [fit_head(feature, residual, e) for e in ends]
    if arm == "random_head":
        i = np.random.default_rng([seed,len(y),635]).integers(len(ends))
    elif arm == "latest" or len(ends)==1:
        i = 0
    else:
        u = (z-lo)/(hi-lo)
        k = kernel(u[:stop],u[:stop]) + .004*np.eye(stop)
        w = np.linalg.solve(k,kernel(u[stop:],u[:stop]).T).T
        # Each snapshot uses the same prefix-only GP and the same four unseen labels.
        scale = np.maximum(net.scale,.02)
        scores = []
        for head in heads:
            delta = feature @ head
            pred = base[stop:] + delta[stop:] + w @ (residual[:stop]-delta[:stop])
            scores.append(losses(arm,pred,y[stop:],goal,scale))
        i = min(range(len(ends)),key=scores.__getitem__)
    return heads[i], ends[i]


def selector(net, goal, log=None):
    def choose(arm, hist, labels, lo, hi, seed, unused):
        rng = np.random.default_rng([seed,len(hist),6481])
        if len(hist)<=4:
            return rng.uniform(lo,hi)
        z = np.asarray(hist)
        y = np.asarray(labels)
        if arm=="mlp200":
            return rank_candidates(z,y,lo,hi,rng,network=net)
        head, end = select_head(arm,net,z,y,lo,hi,goal,seed)
        if log is not None:
            log.append(dict(observed=len(y), trained=end))
        def calibrated(obs,candidates,values,k):
            def predict(points):
                features = net.encode(points)
                return values[0] + net.readout(features) + features @ head
            return predict(obs), predict(candidates)
        return rank_candidates(z,y,lo,hi,rng,calibrated=calibrated)
    return choose
