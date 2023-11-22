import numpy as np 



# Modelling Fourier filtering
def fourier_filter(mask, aperture_position, aperture_radius, get_ft=False):
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


def lowpass_filter(mask,  aperture_radius):
    nr, nc = mask.shape
    ft_mask = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
    fx, fy = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nc)),np.fft.fftshift(np.fft.fftfreq(nr)))
    aperture = fx**2 + fy**2 < aperture_radius**2
    lowpass_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aperture*ft_mask)))

    return lowpass_field, ft_mask