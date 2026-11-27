"""Online monitor using a particle mixture over droop location, size and duration."""
import numpy as np


class ParticleMonitor:
    """Lane descriptors follow physical order. There are no learned lane IDs."""

    def __init__(self, q, amplitude, left, right, batch, rng, particles=96,
                 probe="eye", center=True, ar=.97, noise=.025**2, samples=16):
        self.rng, self.probe, self.center = rng, probe, center
        self.q, self.amplitude = q, amplitude
        self.left, self.right = left, right
        self.ar, self.noise, self.samples = ar, noise, samples
        self.b, self.p, self.n = batch, particles, len(left)
        self.rows = np.arange(batch)
        shape = (batch, particles)
        self.start = rng.integers(self.n, size=shape)
        self.level = np.zeros(shape)
        self.hazard = rng.choice([1/12, 1/32], size=shape)
        self.boundary = rng.integers(max(1, round(self.n/4)), size=shape)
        self.rate = rng.choice([.25, .45], size=shape)
        self.base = np.zeros((*shape, self.n))
        self.shift = np.zeros_like(self.base)
        self.logw = np.full(shape, -np.log(particles))
        self.cov = np.zeros((batch, self.n, self.n))

    def _condition(self, prediction, observation, column, variance):
        error = observation[:, None] - prediction
        self.logw -= .5 * error**2 / variance[:, None]
        self.logw -= np.logaddexp.reduce(self.logw, axis=1)[:, None]
        self.base += (error / variance[:, None])[:, :, None] * column[:, None, :]
        self.cov -= column[:, :, None] * column[:, None, :] / variance[:, None, None]

    def _resample(self):
        w = np.exp(self.logw)
        ix = np.flatnonzero((w*w).sum(axis=1) > 2 / self.p)
        points = (np.arange(self.p) + self.rng.random((len(ix), 1))) / self.p
        parents = np.array([np.searchsorted(row.cumsum(), pts).clip(0, self.p-1)
                            for row, pts in zip(w[ix], points)], dtype=int).reshape(-1, self.p)
        for array in (self.start, self.level, self.hazard, self.boundary,
                      self.rate, self.base, self.shift):
            array[ix] = array[ix[:, None], parents]
        self.logw[ix] = -np.log(self.p)

    def advance(self, common):
        rng, shape = self.rng, self.level.shape
        self._resample()
        switch = rng.random(shape) < self.hazard
        # Quiet is a persistent hypothesis. Rejuvenate boundaries and duration at events.
        self.hazard = np.where(switch & (rng.random(shape) < .2),
                               rng.choice([1/12, 1/32], size=shape), self.hazard)
        width = max(1, round(self.n/4))
        self.boundary = np.where(switch & (rng.random(shape) < .15),
                                 rng.integers(width, size=shape), self.boundary)
        groups = int(np.ceil(self.n / width))
        start = (self.boundary + width*rng.integers(groups, size=shape)) % self.n
        self.start = np.where(switch, start, self.start)
        amp = rng.uniform(.6, 1.4, size=shape) * (rng.random(shape) > .15)
        self.level = np.where(switch, amp, self.level)
        active = (np.arange(self.n) - self.start[:, :, None]) % self.n < width
        self.shift += self.rate[:, :, None] * (
            active*self.level[:, :, None]*self.amplitude - self.shift)
        self.base *= self.ar
        self.cov = self.ar**2*self.cov + self.q
        column = self.cov.mean(axis=2)
        variance = column.mean(axis=1) + self.noise
        self._condition((self.base+self.shift).mean(axis=2), common, column, variance)

    def choose(self):
        modes = self.base+self.shift
        w = np.exp(self.logw)[:, :, None]
        delta = modes - (w*modes).sum(axis=1, keepdims=True)
        total = self.cov + delta.swapaxes(1, 2) @ (w*delta)
        var = total.diagonal(axis1=1, axis2=2)
        reduction = total**2 / (var[:, None, :] + self.noise)
        if self.probe == "eye":
            # Sensitivity of a two-sided Gaussian escape probability to variance.
            half = .5*(self.left+self.right)
            v = np.maximum(var + (1/self.samples)**2/12, 1e-10)
            weight = half*np.exp(-.5*half**2/v)/v**1.5
            reduction *= weight[:, :, None]
        return reduction.sum(axis=1).argmax(axis=1)

    def observe(self, action, reading):
        column = self.cov[self.rows, :, action]
        variance = column[self.rows, action] + self.noise
        prediction = (self.base+self.shift)[self.rows, :, action]
        self._condition(prediction, reading, column, variance)

    def codes(self):
        prediction = (np.exp(self.logw)[:, :, None]*(self.base+self.shift)).sum(axis=1)
        if self.center:
            prediction += .5*(self.right-self.left)
        return np.rint(prediction*self.samples)
