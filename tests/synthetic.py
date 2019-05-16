#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

def triangle(x):
    return np.vectorize(lambda x: 0 if abs(x) > 1 else 1-x if x >= 0 else x+1, otypes=[np.float64])(x)

def semicircle(x):
    r = math.sqrt(2/math.pi)
    return np.vectorize(lambda x: 0 if abs(x) > r else math.sqrt(r**2-x**2), otypes=[np.float64])(x)

def sample_pdf(f, size, xmin = -1, xmax = 1):
    n = 0
    max_x = scipy.optimize.fmin(lambda x: -f(x), 0)
    fmax = f(max_x)

    results = []
    while n < size:
        xtest = np.random.uniform(low=xmin, high=xmax)
        ytest = np.random.uniform(low=0, high=fmax)
        if ytest < f(xtest):
            results.append(xtest)
            n += 1

    return results


# def triangle(x):
#     return (lambda x: 0 if abs(x) > 1 else 1-x if x >= 0 else x+1)(x)

# def semicircle(x):
#     r = math.sqrt(2/math.pi)
#     return (lambda x: 0 if abs(x) > r else math.sqrt(r**2-x**2))(x)

class rv_mixture_1(scipy.stats.rv_continuous):

    def _pdf(self, x):
        return m1(x)
    # def _pdf(self, x, function):
    #     return function(x)


if __name__ == '__main__':
    x_min = -1.5
    x_max = 1.5
    nbins = 40
    x = np.linspace(x_min,x_max,nbins+1)
    bin_size = (x_max-x_min)/nbins
    
    # Shifting the trangle to the left by 0.4 to reveal the tails.
    m1 = lambda x: 0.8*triangle(x+0.4)+0.2*semicircle(x)
    m2 = lambda x: 0.3*triangle(x+0.4)+0.7*semicircle(x)

    nsamples = 100000
    rv1_samples = sample_pdf(m1, nsamples, x_min, x_max)
    rv2_samples = sample_pdf(m2, nsamples, x_min, x_max)
    plt.figure(1)
    plt.plot(x, nsamples*bin_size*m1(x))
    plt.hist(x=rv1_samples, bins=x)
    plt.figure(2)
    plt.plot(x, nsamples*bin_size*m2(x))
    plt.hist(x=rv2_samples, bins=x)
    plt.show()