import numpy as np

def generate_lut(holo, n_SP, step=0.01, save_path=None):
    """Generates the look up table for the Haskell and superpixel holograms.
    
    Parameters
    ----------
    holo : {'haskell', 'haskell45', 'sp'}
        Type of hologram for which to generate the look up table. 
    n_SP : int
        Size of the superpixels.
    step : float, optional 
        Step size used to construct the look up table.
        It determines the minimum difference to consider to complex values as different.
        Note that the lut is constructed assuming the real and imaginary parts of the 
        complex values lie between -1 and +1.
    save_path : str, optional
        If provided it saves the output into a numpy array at the specified location.
        
    Returns
    -------
    float
        Diffraction efficiency.
    """
    # pixels within SP
    m_SP = n_SP**2
    pix_ind_list = np.arange(m_SP)
    # complex value assigned to each pixel 
    if holo=='haskell':
        unit_points = np.exp(1j*2*np.pi*pix_ind_list/n_SP) 
    elif holo=='haskell45':
        unit_points = np.zeros(m_SP, complex)
        for i in range(n_SP):
            unit_points[n_SP*i:n_SP*(i+1)] = np.exp(1j*2*np.pi*(pix_ind_list[n_SP*i:n_SP*(i+1)]+i)/n_SP)
    elif holo=='sp':
        unit_points = np.exp(1j*2*np.pi*pix_ind_list/m_SP)   
    else:
        raise ValueError('Invalid option for holo. Choose between haskell, haskel45, sp.')
    field_values, unique_pixel_combinations, lut = _build_lut_from_unit_pts(unit_points, step)


    if save_path is not None:
        np.savez(save_path+holo+'_lut'+str(n_SP), 
                field_values=field_values, 
                unique_pixel_combinations=unique_pixel_combinations, 
                lut=lut, step=[step])
    
    return field_values, unique_pixel_combinations, lut

def _build_lut_from_unit_pts(unit_points, step):

    m_SP = len(unit_points)
    # Compute all possible field combinations and the corresponding pixel arrangement
    field_combinations = np.zeros(2**m_SP,dtype=complex)
    pixel_combinations = np.zeros((2**m_SP,m_SP),dtype=int)
    numbers = np.arange(2**m_SP)
    for pix_ind in range(m_SP):
        field_combinations += np.mod(numbers,2)*unit_points[pix_ind]
        pixel_combinations[:,pix_ind] = np.mod(numbers,2)
        numbers = numbers//2
    field_combinations = np.round(field_combinations,10)
    # Sort according to field values
    sort_ind = np.argsort(field_combinations)
    pixel_combinations = pixel_combinations[sort_ind]
    field_combinations = field_combinations[sort_ind]
    # Find unique combinations
    field_values, field_ind, field_counts = \
        np.unique(field_combinations, return_index=True,
                  return_counts=True)
    n_field = len(field_values)
    unique_pixel_combinations = np.zeros((n_field,m_SP),dtype=int)
    for ind in range(n_field):
        # choose the one with less pixels turned on
        pix_sum_minind = np.argmin(np.sum(pixel_combinations[field_ind[ind]
            :field_ind[ind]+field_counts[ind]],axis=1))
        unique_pixel_combinations[ind] = pixel_combinations[field_ind[ind]+pix_sum_minind]
    field_values /= np.max(np.abs(field_values))
    # Create LUT with a defined step size 
    ri_range = np.arange(-1,1+step,step)
    imag_grid, real_grid = np.meshgrid(ri_range,ri_range)
    complex_grid = real_grid +1j*imag_grid
    n_amp_range = len(ri_range)
    lut = np.zeros_like(complex_grid,dtype=int)
    for re in range(n_amp_range):
        for im in range(n_amp_range):
            ind = np.argmin(np.abs(complex_grid[re,im]-field_values))
            lut[re,im] = ind

    return field_values, unique_pixel_combinations, lut



# def generate_Haskell_lut(n_SP, step=0.01, save_path=None):
#     # pixels within SP
#     m_SP = n_SP**2
#     pix_ind_list = np.arange(m_SP)
#     # complex value assigned to each pixel 
#     unit_points = np.exp(1j*2*np.pi*pix_ind_list/n_SP)   
    
#     field_values, unique_pixel_combinations, lut = build_lut_from_unit_pts(unit_points, step)

#     if save_path is not None:
#         np.savez(save_path+'haskell_lut'+str(n_SP), 
#                 field_values=field_values, 
#                 unique_pixel_combinations=unique_pixel_combinations, 
#                 lut=lut, step=[step])
    
#     return field_values, unique_pixel_combinations, lut

# def generate_SP_lut(n_SP, step=0.01, save_path=None):
#     # pixels within SP
#     m_SP = n_SP**2
#     pix_ind_list = np.arange(m_SP)
#     # complex value assigned to each pixel 
#     unit_points = np.exp(1j*2*np.pi*pix_ind_list/m_SP)   

#     field_values, unique_pixel_combinations, lut = build_lut_from_unit_pts(unit_points, step)

#     if save_path is not None:
#         np.savez(save_path+'sp_lut'+str(n_SP), 
#                 field_values=field_values, 
#                 unique_pixel_combinations=unique_pixel_combinations, 
#                 lut=lut, step=[step])
    
#     return field_values, unique_pixel_combinations, lut
