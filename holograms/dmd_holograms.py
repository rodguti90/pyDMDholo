import numpy as np 



#######################################################################
# Aux definition
#######################################################################

def holo_efficiency(nuvec):
    px= 1/nuvec[0]
    eta_dif = np.sin(np.pi*np.floor(px/2)/px)/np.pi
    if nuvec[1]!=0:
        py = 1/nuvec[1]
        eta_dif *= px*np.sin(np.pi*px/py)/np.pi
    return eta_dif

def holo_preamble(field, nuvec, renorm=True):
    a = np.abs(field)
    phi = np.angle(field)
    sh = field.shape
    x,y = np.meshgrid(np.arange(sh[1]),np.arange(sh[0]))
    nunorm = 2*np.linalg.norm(nuvec)
    if renorm:
        a /=np.max(a)
    return a, phi, x-nuvec[1]/nunorm, y+nuvec[0]/nunorm

#######################################################################
# Gray-scale hologram definition
#######################################################################
def amplitude_off_axis(field, nuvec=(1/4,0)):
    a, phi, x, y = holo_preamble(field, nuvec)
    return 1/2 + a* np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)/2

def amplitude_lee(field, nuvec=(1/4,0)):
    a, phi, x, y = holo_preamble(field, nuvec)
    return a* (np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi) 
               + np.abs(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)))

#######################################################################
# Lee hologram definition
#######################################################################

def parallel_lee(field, nuvec=(1/4,0), renorm=True):
    a, phi, x, y = holo_preamble(field, nuvec, renorm=renorm)
    return np.abs(np.mod((nuvec[0]*x + nuvec[1]*y)-phi/(2*np.pi)-1/2, 1) -1/2)\
        < np.arcsin(a)/(2*np.pi)

def orthogonal_lee(field, nuvec=(1/4,0), renorm=True):
    a, phi, x, y = holo_preamble(field, nuvec, renorm=renorm)
    
    return (np.abs(np.mod((nuvec[0]*x + nuvec[1]*y)-phi/(2*np.pi)-1/2, 1) -1/2-0*1/2) < 1/4) \
            *(np.mod((-nuvec[1]*x + nuvec[0]*y), 1) < 1*a)

#######################################################################
# Look-up table hologram definition
#######################################################################

def down_sample(field, nsp, method='center'):
    if method=='center':
        ds_field = field[nsp//2::nsp,nsp//2::nsp]
    elif method=='mean':
        ds_field = np.zeros_like(field[nsp//2::nsp,nsp//2::nsp])
        for i in range(nsp):
            for j in range(nsp):
                ds_field += field[i::nsp,j::nsp]
        ds_field /= nsp**2
    elif method=='side':
        ds_field = field[::nsp,::nsp]
    else:
        raise ValueError('Invalid option for method.')
    return ds_field


def holo_Haskell(us_field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    if renorm:
        us_field /= np.max(np.abs(us_field))
    # assume the field has been rescaled to the unit SP
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))
    
    field = down_sample(us_field, n_SP, method=ds_method)
    sh = field.shape
    holo = np.zeros((sh[-2]*n_SP, sh[-1]*n_SP),dtype=int)
    # rescale field values according to LUT
    field_sc = field/(np.max(np.abs(field))*step)
    reim0 = (len(lut))//2
    for j in range(sh[-2]):
        for i in range(sh[-1]):
            re = int(np.round(np.real(field_sc[j,i])))
            im = int(np.round(np.imag(field_sc[j,i])))
            # sp_pixel = np.roll(pixel_combinations[lut[re+reim0, im+reim0]], -shift)
            sp_pixel = pixel_combinations[lut[re+reim0, im+reim0]]
            holo[n_SP*j:n_SP*(j+1),n_SP*i:n_SP*(i+1)] = sp_pixel.reshape(n_SP,n_SP)
    return holo

def holo_SP(us_field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    if renorm:
        us_field /= np.max(np.abs(us_field))
    # assume the field has been rescaled to the unit SP
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))

    field = down_sample(us_field, n_SP, method=ds_method)
    sh = field.shape
    holo = np.zeros((sh[-2]*n_SP, sh[-1]*n_SP),dtype=int)
    # rescale field values according to LUT
    field_sc = field/(np.max(np.abs(field))*step)
    reim0 = (len(lut))//2
    for j in range(sh[-2]):
        shift = np.mod(n_SP*j, n_SP**2)
        for i in range(sh[-1]):
            re = int(np.round(np.real(field_sc[j,i])))
            im = int(np.round(np.imag(field_sc[j,i])))
            sp_pixel = np.roll(pixel_combinations[lut[re+reim0, im+reim0]], -shift)
            holo[n_SP*j:n_SP*(j+1),n_SP*i:n_SP*(i+1)] = sp_pixel.reshape(n_SP,n_SP).T
    return holo



# def parallel_lee2(field, nuvec=(1/4,0)):
#     a, phi, x, y = holo_preamble(field, nuvec)
#     return (1+np.sign(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi) - np.cos(np.arcsin(a))))/2

# def orthogonal_lee2(field, nuvec=(1/4,0), renorm=True):
#     a, phi, x, y = holo_preamble(field, nuvec, renorm=renorm)
    
#     return (1+np.sign(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)))/2 \
#         * (1+np.sign(np.cos(2*np.pi*(-nuvec[1]*x + nuvec[0]*y)) - np.cos(np.pi*a)))/2
# Constellations
