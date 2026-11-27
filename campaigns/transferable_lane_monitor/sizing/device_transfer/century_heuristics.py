"""Sequential ACOR, DE/rand/1/bin and sampling controls in log-width coordinates."""
import numpy as np
from alarm_model import cost
from experiment import pool, upper
from proposal import propose


class Heuristic:
    def __init__(self, arm, circuit, seed, bo_goal=.90):
        self.bo_goal = bo_goal
        self.arm = arm
        self.c = circuit
        self.rng = np.random.default_rng(seed)
        self.hi = np.log(upper(circuit))
        self.lo = np.full_like(self.hi, np.log(.58))
        self.term = np.repeat([j in circuit.outputs for j in range(len(circuit.parents))], 2)
        self.grouped = arm in ("group_acor", "group_random")
        if self.grouped:
            # Four factors: internal N/P and terminal N/P, legal for every member.
            self.group = 2*self.term.astype(int) + np.arange(len(self.hi)) % 2
            self.hi = np.array([self.hi[self.group == i].min() for i in range(4)])
            self.lo = np.full(4, np.log(.58))
        self.archive = []
        self.target = 0

    def encode(self, x):
        if self.grouped:
            return np.array([np.log(x[self.group == i]).mean() for i in range(4)])
        return np.log(x)

    def decode(self, z):
        return np.exp(z[self.group] if self.grouped else z)

    def ask(self, hist, labels):
        t = len(hist)-1
        if t == 0:
            return np.where(self.term, .84, 1.)
        if self.arm == "bo":
            samples = pool(self.c, self.rng, hist[np.argmin(cost(labels))])
            return propose("cold_bo", {}, self.c, hist, labels, samples, self.rng, goal=self.bo_goal)
        # Six members including the common nominal and electrical-role probe.
        if self.arm in ("random", "group_random") or len(hist) < 6:
            return self.decode(self.rng.uniform(self.lo, self.hi))
        if self.arm == "de":
            if not self.archive:
                self.archive = list(range(6))
            else:
                parent = self.archive[self.target]
                if cost(labels[-1]) <= cost(labels[parent]):
                    self.archive[self.target] = len(hist)-1
                self.target = (self.target+1) % 6
            ids = self.rng.choice(np.delete(self.archive, self.target), 3, replace=False)
            a, b, c = np.log(hist[ids])
            donor = a + .65*(b-c)
            mask = self.rng.random(len(a)) < .8
            mask[self.rng.integers(len(a))] = True
            z = np.where(mask, donor, np.log(hist[self.archive[self.target]]))
        else:
            elite = np.argsort(cost(labels), kind="stable")[:6]
            population = np.stack([self.encode(hist[j]) for j in elite])
            weight = np.exp(-.5*(np.arange(6)/(.40*6))**2)
            anchor = population[self.rng.choice(6, p=weight/weight.sum())]
            sigma = .85*np.abs(population-anchor).sum(0)/5
            z = self.rng.normal(anchor, sigma)
        # Reflection keeps mutation legal without accumulating boundary point masses.
        span = self.hi-self.lo
        z = self.lo + span - np.abs(np.mod(z-self.lo, 2*span)-span)
        return self.decode(z)
