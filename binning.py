import numpy as np
from lens import Lens

name = 'abell2744_mass_density'
abell = Lens(f'data/{name}.fits')

nr, kbin_centers, power = abell.compute_power_spectrum(kbin_method='experimental')
theta_bin_centers = 1.0/kbin_centers
sig_power = power/(np.sqrt(0.5*nr))

print(sig_power)
abell.print_bin_info()
