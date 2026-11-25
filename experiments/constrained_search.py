"""Online constrained Bayesian optimization on a normalized design box.

Observe a positive objective to minimize and signed quality residuals (<= 0
passes). Missing simulation results consume budget without becoming targets.
The optimizer uses no circuit-specific rules or offline training data.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import ndtr


def matern(a, b, length):
    distance = np.sqrt(5.)*cdist(a/length,b/length)
    return (1+distance+distance**2/3)*np.exp(-distance)


class Surrogate:
    """A shared kernel with per-output scaling and a regularized linear trend."""

    def __init__(self, x, y, radius):
        self.x = x
        self.location = np.median(y,axis=0)
        self.scale = np.maximum(np.std(y,axis=0), .04)
        outputs = np.clip((y-self.location)/self.scale, -5, 5)
        trend = np.c_[np.ones(len(x)),x-.5]
        self.beta = np.linalg.solve(trend.T@trend+np.diag([.1,*([2.]*x.shape[1])]),
                                    trend.T@outputs)
        residual = outputs-trend@self.beta
        self.length = max(.16, .42*radius/.35)
        covariance = matern(x,x,self.length)+np.eye(len(x))*.008
        self.factor = cho_factor(covariance,lower=True)
        self.alpha = cho_solve(self.factor,residual)
        self.amplitude = np.maximum(.25,np.sqrt(np.mean(residual*self.alpha,axis=0)))

    def predict(self, x):
        cross = matern(self.x,x,self.length)
        trend = np.c_[np.ones(len(x)),x-.5]
        mean = self.location + self.scale*(trend@self.beta+cross.T@self.alpha)
        solved = solve_triangular(self.factor[0],cross,lower=True)
        deviation = np.sqrt(np.maximum(.006, 1-np.sum(solved**2,axis=0)))
        return mean, deviation[:,None]*self.scale*self.amplitude


class ConstrainedSearch:
    def __init__(self, dimension, constraint_count, seed):
        if dimension < 1 or constraint_count < 1:
            raise ValueError('Expected positive numbers of variables and constraints')
        self.dimension = dimension
        self.constraint_count = constraint_count
        self.rng = np.random.default_rng(seed)
        self.points = []
        self.results = []
        self.radius = .32
        self.successes = 0
        self.failures = 0
        self.pending = None

    @staticmethod
    def rank(row):
        if row is None:
            return np.inf,np.inf
        return max(0.,float(np.max(row[1:]))),float(row[0])

    def observe(self, point, objective=None, constraints=None):
        point = np.asarray(point,dtype=float)
        if point.shape != (self.dimension,) or not np.isfinite(point).all() or np.any((point<0)|(point>1)):
            raise ValueError('Expected a finite normalized design')
        active = self.pending is not None
        if active and not np.array_equal(point,self.pending):
            raise ValueError('Observation differs from the outstanding proposal')
        row = None
        if objective is not None or constraints is not None:
            constraints = np.asarray(constraints,dtype=float)
            if objective is None or not np.isfinite(objective) or objective<=0 or constraints.shape!=(self.constraint_count,) or not np.isfinite(constraints).all():
                raise ValueError('Expected positive finite objective and finite constraint vector')
            row = np.r_[np.log(objective),constraints]
        previous = min(map(self.rank,self.results),default=(np.inf,np.inf))
        self.points.append(point.copy())
        self.results.append(row)
        if active:
            if self.rank(row)<previous:
                self.successes += 1
                self.failures = 0
                if self.successes >= 2:
                    self.radius = min(.6,1.28*self.radius)
                    self.successes = 0
            else:
                self.successes = 0
                self.failures += 1
                if self.failures >= 3:
                    self.radius = .32 if self.radius<.085 else .65*self.radius
                    self.failures = 0
        self.pending = None

    def ask(self):
        if self.pending is not None:
            raise RuntimeError('Observe or reject the outstanding proposal first')
        d = self.dimension
        history = np.asarray(self.points).reshape(-1,d)
        valid = [i for i,r in enumerate(self.results) if r is not None]
        global_points = self.rng.uniform(0,1,(512,d))
        if len(valid) < 3:
            self.pending = global_points[0].copy()
            return self.pending.copy()

        ordered = sorted(valid,key=lambda i:self.rank(self.results[i]))
        elite = history[ordered[:3]]
        best = self.results[ordered[0]]
        base = elite[self.rng.choice(len(elite),3072,p=np.array([.72,.19,.09])[:len(elite)]/
                                     np.sum([.72,.19,.09][:len(elite)]))]
        mask = self.rng.uniform(size=base.shape)<min(1.,3/d)
        mask[np.arange(len(mask)),self.rng.integers(d,size=len(mask))] = True
        step = self.rng.uniform(-1,1,base.shape)*self.radius
        local = np.clip(base+mask*step,0,1)
        axes = np.eye(d)*self.radius
        coordinate = np.clip(elite[0]+np.concatenate([axes,-axes,.2*axes,-.2*axes]),0,1)
        candidates = np.vstack([local,coordinate,global_points])
        separation = cdist(candidates,history,metric='chebyshev').min(axis=1)
        candidates = candidates[separation>1e-5]

        x = history[valid]
        y = np.asarray([self.results[i] for i in valid])
        model = Surrogate(x,y,self.radius)
        mean, std = model.predict(candidates)
        feasible = ndtr(-mean[:,1:]/std[:,1:]).clip(1e-10,1.)
        log_prob = .7*np.log(feasible.min(axis=1))+.3*np.log(feasible).mean(axis=1)
        if self.rank(best)[0] == 0:
            gain = best[0]-mean[:,0]
            z = gain/std[:,0]
            improvement = gain*ndtr(z)+std[:,0]*np.exp(-z*z/2)/np.sqrt(2*np.pi)
            score = np.log(np.maximum(improvement,1e-12))+log_prob
        else:
            optimistic_violation = np.max(mean[:,1:]-.65*std[:,1:],axis=1)
            score = -optimistic_violation+.12*log_prob
        self.pending = candidates[np.argmax(score)].copy()
        return self.pending.copy()
