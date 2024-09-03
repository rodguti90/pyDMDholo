import numpy as np 



# Modelling Fourier filtering
def fourier_filter(mask, aperture_position, aperture_radius, get_ft=False):
    """Simulates the shaping of fields using hologram mask.
    
    Parameters
    ----------
    mask : ndarray
        Input field to filter.
    aperture_position : float
        Position of the circular aperture used to filter the diffraction order.
    aperture_radius : float
        Radius of the circular aperture used to filter the diffraction order. 
    get_ft : bool, optional 
        If true, it outputs the fourier transform of mask without recentering the desired order.
        
    Returns
    -------
    ndarray
        Array containing the shaped field.
    """
    nr, nc = mask.shape
    Col, Row = np.meshgrid(np.arange(nc),np.arange(nr))
    phase_shift = np.exp(1j*2*np.pi* (aperture_position[0]*Col+aperture_position[1]*Row))
    ft_mask = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase_shift*mask)))
    fx, fy = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nc)),np.fft.fftshift(np.fft.fftfreq(nr)))
    aperture = fx**2 + fy**2 < aperture_radius**2
    filtered_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aperture*ft_mask)))
    if get_ft:
        ft_mask = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
    return filtered_field, ft_mask


def lowpass_filter(field, aperture_radius):
    """Computes the lowpass filtered version of field.
    
    It creates a random superposition of plane wave with a Gausian
    eveloppe.
    
    Parameters
    ----------
    field : ndarray
        Input field to filter.
    aperture_radius : float
        Radius of the circular aperture used to fileter the field. 

    Returns
    -------
    ndarray
        Array containing the filtered field.
    """
    nr, nc = field.shape
    ft_mask = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    fx, fy = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nc)),np.fft.fftshift(np.fft.fftfreq(nr)))
    aperture = fx**2 + fy**2 < aperture_radius**2
    lowpass_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aperture*ft_mask)))

    return lowpass_field, ft_mask