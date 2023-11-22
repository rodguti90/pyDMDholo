import numpy as np
from numpy.polynomial.hermite import hermval2d
from scipy.special import factorial
from numpy.random import uniform

from .functions import rnd_cpx_disk, rnd_disk



# Field definitions


def speckle_gauss(npw, kmax, x, y):
    r2 = x**2 + y**2
    ks = rnd_disk(size=npw)*kmax
    cs = rnd_cpx_disk(size=npw)
    sg = np.sum(cs[:,None,None]*np.exp(1j*2*np.pi*(ks[0,:,None,None]*x+ks[1,:,None,None]*y)), axis=0)
    return sg*np.exp(-r2)

def laguerre_gauss(N, ell, x, y):
    r2 = x**2 + y**2

    c = (1j**(np.abs(ell)-N))*np.sqrt(2**(np.abs(ell)+1) * factorial((N-np.abs(ell))//2) 
                 / (np.pi * factorial((N+np.abs(ell))//2)))
    vortex = (x+np.sign(ell)*1j*y)**np.abs(ell)
    lg = c * vortex * laguerre_pol((N-np.abs(ell))//2,np.abs(ell),2*r2) *np.exp(-r2)
    return lg

def hermite_gauss(N, ell, x, y):
    r2 = x**2 + y**2
    c = 1/np.sqrt(np.pi * 2**(N-1) * factorial((N-np.abs(ell))//2) 
                  * factorial((N+np.abs(ell))//2))
    
    hg = c* hermite_pol((N+np.abs(ell))//2, 2**(1/2) * x) \
        * hermite_pol((N-np.abs(ell))//2, 2**(1/2) * y) *np.exp(-r2)

    return hg 

def laguerre_pol(p,ell,x):
    xr=x.ravel()
    Lplt=np.zeros((len(xr),np.max([p+1,2])))
    Lplt[:,0]=1
    Lplt[:,1]=-xr+ell+1
    for ii in range(1,p):
        Lplt[:,ii+1]=((2*ii+ell+1-xr)*Lplt[:,ii]-(ii+ell)*Lplt[:,ii-1])/(ii+1)
    
    return np.reshape(Lplt[:,p], x.shape)

def hermite_pol(n,x):
    xr=x.ravel()
    h=np.zeros((len(xr),np.max([n+1,2])))
    h[:,0]=1
    h[:,1]=2*xr
    for ii in range(1,n):
        h[:,ii+1] = 2*xr*h[:,ii] -2*(ii)*h[:,ii-1]

    return np.reshape(h[:,n], x.shape)