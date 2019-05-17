#!/usr/bin/env python

from scipy.stats import rv_continuous

import math

class triangle(rv_continuous):
    """Triangular distribution."""
    def _pdf(self, x):
        return (lambda x: 0 if abs(x) > 1 else 1-x if x >= 0 else x+1)(x)
    # If cdf is not manually given, rvs() won't work.
    def _cdf(self, x):
        return (lambda x: 0 if x < -1 else .5*x*x+x+.5 if x < 0 else -.5*x*x+x+.5 if x < 1 else 1)(x)

class semicircle(rv_continuous):
    """Semicircle distribution."""
    def _pdf(self, x):
        r = math.sqrt(2/math.pi)
        return (lambda x: 0 if abs(x) > r else math.sqrt(r**2-x**2))(x)


if __name__ == '__main__':
    tri_dist = triangle(name='tri_dist')
    semic_dist = semicircle(name='semic_dist')
    print(tri_dist.rvs(size=10), semic_dist.rvs(size=10))