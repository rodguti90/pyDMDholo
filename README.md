# pyDMDholo

© Rodrigo Gutiérrez Cuevas

pyDMDholo provides the necessary python code and example for writing binary hologram for shaping light with digital micro mirror devices (DMDs), as described in the following paper:

- Binary holograms for wavefront shaping with digital micromirror devices. R. Gutiérrez-Cuevas, and S.M. Popoff, [*arXiv:0000.00000 [physics.optics]* (2023)](https://doi.org/10.00000/arXiv.0000.00000).

## What is it?

pyDMDholo provides all the necessary python code to generate binary holograms that shape an incoming plane wave into a target complex field. It implements the most commonly used Lee holograms as well as the superpixel method. 
It also includes the predecessor to the superpixel method referred to as the Haskell hologram, as well as variations of it. Furthermore, it also includes functions to generate Hermite-Gauss and Laguerre-Gauss beams, and speckle fields. Likewise, it provides the necessary functions to model the propagation of the reflected light from the hologram through the Fourier filter in a 4f configuration. Examples of use are also provided.

## How does it work?

[/holograms](holograms/): Functions for generating holograms and necessary look-up tables.

[/fields_propagation](fields_propagation/): Functions for generation LG, HG and speckle fields and for simulating the Fourier filtering for selecting the appropriate order.

[/plotting](plotting/): Assortment of plotting functions used in the examples. 

### Examples of use

1. [`Example_of_use.ipynb`](Example_of_use.ipynb): Provides a simple example for shaping a Laguerre-Gauss beam with the different types of holograms.
2. [`Quantization_complex_values.ipynb`](Quantization_complex_values.ipynb): Showcases the quantization of the complex values that can be encoded into a given hologram due to the pixelization of the DMD.
3. [`Comparing_shaped_fields.ipynb`](Comparing_shaped_fields.ipynb): Compares the shaping of LG, HG, and speckle fields using various types of holograms in different configurations.

## Citing the code

If the code was helpful for your work, please consider citing it along with the accompanying paper.
