#!/usr/bin/env python

from scipy.stats import rv_continuous

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def triangle_pdf(x):
    return (lambda x: 0 if abs(x) > 1 else 1-x if x >= 0 else x+1)(x)

def triangle_cdf(x):
    return (lambda x: 0 if x < -1 else .5*x*x+x+.5 if x < 0 else -.5*x*x+x+.5 if x < 1 else 1)(x)

def semicircle_pdf(r,a,x):
    return (lambda x: 0 if abs(x-a) > r else math.sqrt(r**2-(x-a)**2)*2/math.pi/r**2)(x)

def semicircle_cdf(r,a,x):
    """
    I used sympy to get the indefinite integral and utilized
    the fundamental theorem of calculus to get the answer.

    In [1]: from sympy import *
    In [2]: r,x,a=symbols('r x a')
    In [3]: init_printing(use_unicode=False, wrap_line=False, no_global=True)
    In [4]: integrate(sqrt(r**2-(x-a)**2), x)
    Out[4]: 
    /     2      /-a + x\                                                                            
    |  I*r *acosh|------|                                               3              |       2|    
    |            \  r   /         I*r*(-a + x)                I*(-a + x)               |(a - x) |    
    |- ------------------ - ------------------------ + --------------------------  for |--------| > 1
    |          2                    ________________             ________________      |    2   |    
    |                              /              2             /              2       |   r    |    
    |                             /       (-a + x)             /       (-a + x)                      
    |                       2*   /   -1 + ---------    2*r*   /   -1 + ---------                     
    |                           /              2             /              2                        
    <                         \/              r            \/              r                         
    |                                                                                                
    |                                       _______________                                          
    |                                      /             2                                           
    |                                     /      (-a + x)                                            
    |              2     /-a + x\   r*   /   1 - --------- *(-a + x)                                 
    |             r *asin|------|       /             2                                              
    |                    \  r   /     \/             r                                               
    |             --------------- + --------------------------------                   otherwise     
    \                    2                         2
    """
    if x < a-r: return 0
    if x > a+r: return 1
    return 1./math.pi/r**2*(r**2*(math.asin((x-a)/r)-math.asin(-1))+r*(x-a)*math.sqrt(1-(x-a)**2/r**2))

class triangle(rv_continuous):
    """Triangular distribution."""
    def _pdf(self, x):
        return triangle_pdf(x)
    # If cdf is not manually given, rvs() won't work.
    def _cdf(self, x):
        return triangle_cdf(x)

class semicircle(rv_continuous):
    """Semicircle distribution."""
    rr = 1.1
    aa = -0.1
    def _pdf(self, x):
        return semicircle_pdf(self.rr, self.aa, x)
    def _cdf(self, x):
        return semicircle_cdf(self.rr, self.aa, x)

class mixture1(rv_continuous):
    """Mixture distribution."""
    rr = 1.1
    aa = -0.1
    r1 = 0.1
    r2 = 0.9
    def _pdf(self, x):
        return self.r1*semicircle_pdf(self.rr, self.aa, x)+self.r2*triangle_pdf(x)
    def _cdf(self, x):
        return self.r1*semicircle_cdf(self.rr, self.aa, x)+self.r2*triangle_cdf(x)


if __name__ == '__main__':

    # command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nsamples',type=int,default=10000)
    args = parser.parse_args()

    # instantiate my distributions
    tri_dist = triangle(name='tri_dist')
    semic_dist = semicircle(name='semic_dist')
    mix_dist = mixture1(name='mix_dist')

    # make samples
    n_samples = args.nsamples
    s_tri = tri_dist.rvs(size=n_samples)
    s_semic = semic_dist.rvs(size=n_samples)
    s_mix = mix_dist.rvs(size=n_samples)

    # make samples into dataframes
    df_tri = pd.DataFrame(data=s_tri,index=None,columns=['sample_value'])
    df_mix = pd.DataFrame(data=s_mix,index=None,columns=['sample_value'])

    # save dataframes to file
    if not os.path.exists('test_input'):
        os.makedirs('test_input')
    df_tri.to_hdf('test_input/samples_from_distributions.h5', key='triangle_samples', complevel=9)
    df_mix.to_hdf('test_input/samples_from_distributions.h5', key='mixture_samples', complevel=9)

    # visualize the samples
    fig1 = plt.figure(1)
    h_tri, bin_tri, p_tri = plt.hist(s_tri, bins='auto')
    x_tri = np.linspace(bin_tri[0],bin_tri[-1],len(bin_tri))
    plt.plot(x_tri, n_samples*np.diff(bin_tri)[0]*np.vectorize(triangle_pdf)(x_tri))
    fig2 = plt.figure(2)
    h_semic, bin_semic, p_semic = plt.hist(s_semic, bins='auto')
    x_semic = np.linspace(bin_semic[0],bin_semic[-1],len(bin_semic))
    plt.plot(x_semic, n_samples*np.diff(bin_semic)[0]*np.vectorize(semicircle_pdf)(semicircle.rr,semicircle.aa,x_semic))
    fig3 = plt.figure(3)
    h_mix, bin_mix, p_mix = plt.hist(s_mix, bins='auto')
    x_mix = np.linspace(bin_mix[0],bin_mix[-1],len(bin_mix))
    plt.plot(x_mix, n_samples*np.diff(bin_mix)[0]*(mixture1.r1*np.vectorize(semicircle_pdf, otypes=[float])(mixture1.rr,mixture1.aa,x_mix)+mixture1.r2*np.vectorize(triangle_pdf, otypes=[float])(x_mix)))
    
    # save plots to files
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig1.savefig('plots/triangle_sample_hist_{}_entries.pdf'.format(n_samples))
    fig2.savefig('plots/semicircle_sample_hist_{}_entries.pdf'.format(n_samples))
    fig3.savefig('plots/mixture1_sample_hist_{}_entries.pdf'.format(n_samples))