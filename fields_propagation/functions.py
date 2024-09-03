import numpy as np
from numpy.random import uniform
from scipy.stats import pearsonr


def rnd_cpx_disk(size=None):
    """Generates random complex numbers uniformly distributed inside the unit disk.
    
    Parameters
    ----------
    size : int
        Dimension of the returned array.

    Returns
    -------
    ndarray
        Array containing the random numbers
    """
    rho = np.sqrt(uniform(0., 1, size))
    phi = uniform(0, 2*np.pi, size)
    return rho * np.exp(1j*phi)

def rnd_disk(size=None):
    """Generates random 2d points uniformly distributed inside the unit disk.
    
    Parameters
    ----------
    size : int
        Dimension of the returned array.

    Returns
    -------
    ndarray
        Array containing the random numbers
    """
    rho = np.sqrt(uniform(0., 1, size))
    phi = uniform(0, 2*np.pi, size)
    return rho * np.array([ np.cos(phi),np.sin(phi)])

def cpx_corr(Y1,Y2):
    """Compute the complex correlation between two complex arrays.
        
    Parameters
    ----------
    Y1 : ndarray
        First complex array.
    Y2 : ndarray
        Second complex array.

    Returns
    -------
    float
        complex correlation value
    """
    corr = np.sum((Y1)*(Y2).conj())
    norm_fac = np.sqrt(np.sum(np.abs(Y1)**2)*np.sum(np.abs(Y2)**2))  
    return corr/norm_fac

def int_corr(f1, f2):
    """Compute the Pearson correlation between two intensity arrays.
        
    Parameters
    ----------
    Y1 : ndarray
        First complex array.
    Y2 : ndarray
        Second complex array.

    Returns
    -------
    float
        Correlation value
    """
    i1 =np.abs(f1.ravel())**2
    i2 =np.abs(f2.ravel())**2
    return pearsonr(i1,i2)[0]