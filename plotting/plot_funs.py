import numpy as np
from colorsys import hls_to_rgb

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'light' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = np.transpose(c, (1,2,0))  
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c

def zoom_image(image, zoom, shift=[0,0]):
    shift_nd=np.array(shift)
    center = np.array(image.shape[:2])//2 - shift_nd
    z1=(center*(1-1/zoom)).astype(int)
    z2=(center*(1+1/zoom)).astype(int)
    return image[z1[0]:z2[0],z1[1]:z2[1]]