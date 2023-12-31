import numpy as np
from numpy.random import uniform
from scipy.stats import pearsonr


def rnd_cpx_disk(size=None):
    rho = np.sqrt(uniform(0., 1, size))
    phi = uniform(0, 2*np.pi, size)
    return rho * np.exp(1j*phi)

def rnd_disk(size=None):
    rho = np.sqrt(uniform(0., 1, size))
    phi = uniform(0, 2*np.pi, size)
    return rho * np.array([ np.cos(phi),np.sin(phi)])

def cpx_corr(Y1,Y2):
    corr = np.sum((Y1)*(Y2).conj())
    norm_fac = np.sqrt(np.sum(np.abs(Y1)**2)*np.sum(np.abs(Y2)**2))  
    return corr/norm_fac

def int_corr(f1, f2):
    i1 =np.abs(f1.ravel())**2
    i2 =np.abs(f2.ravel())**2
    return pearsonr(i1,i2)[0]