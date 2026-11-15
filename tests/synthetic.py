"""Synthetic phase map from example.ipynb (Section 4 of the elastic k-means paper).

A 12x12 grid on the unit square is split into three regions by two parabolas.
Each region is assigned a sum of 1, 2 or 3 Gaussian peaks whose domain is
randomly warped, so curves from the same phase differ in peak position.

Unlike the notebook, the random draws use a seeded ``numpy.random.RandomState``
instead of the global RNG so that tests are reproducible.
"""
import numpy as np

from autophasemap import BaseDataSet


def phase1(x):
    return 0.5 * x**2 + 0.45


def phase2(x):
    return -0.45 * x**2 + 0.55


def get_label(c):
    """Phase label (1, 2 or 3) of a point ``c`` in the design space."""
    if c[1] - phase1(c[0]) > 0:
        return 1
    elif c[1] - phase2(c[0]) < 0:
        return 2
    return 3


def make_parabolic_design(n_grid=12):
    """Grid of design points and their ground-truth phase labels."""
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    labels = np.array([get_label(p) for p in points])
    return points, labels


class ParabolicPhaseDataSet(BaseDataSet):
    """DataSet from example.ipynb; ``P[i]`` is the number of peaks of sample ``i``."""

    def __init__(self, C, P, n_domain=100, seed=None):
        super().__init__(n_domain=n_domain)
        self.t = np.linspace(0, 1, num=self.n_domain)
        self.N = C.shape[0]
        self.P = P
        self.C = C
        self.n_clusters = 3
        self.rng = np.random.RandomState(seed)

    def g(self, t, p):
        out = np.zeros(self.t.shape)
        for i in range(1, p + 1):
            zi = self.rng.normal(1, 0.1)
            mean = (2 * i - 1) / (2 * p)
            std = 1 / (3 * p)
            out += zi * self.phi(t, mean, std)
        return out

    def phi(self, t, mu, sigma):
        factor = 1 / (2 * (sigma**2))
        return np.exp(-factor * (t - mu) ** 2)

    def gamma(self):
        a = self.rng.uniform(-3, 3)
        if a == 0:
            return self.t
        return (np.exp(a * self.t) - 1) / (np.exp(a) - 1)

    def generate(self):
        self.F = [self.g(self.gamma(), self.P[i]) for i in range(self.N)]
