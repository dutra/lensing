import numpy as np
from numpy import fft as fftengine
from numpy import pi
import scipy.stats as stats
import astropy.io.fits as pyfits
import pylab

import logging
log = logging.getLogger("ps")

class Lens(object):
    def __init__(self, input_map):
        self.input_map_filepath = input_map

        self.load_file()
        self.calculate_field_size()

    def load_file(self):
        # read convergence map from file:
        input_map, self.header = pyfits.getdata(self.input_map_filepath, header=True)
        # The 'meanmass.fits' is a surface density map (M_sun/kpc^2)
        # in units of 10^12 M_sun
        # To make it dimensionless divide by Sigma_crit=(c^2/4piG)*D_os*D_ol/D_ls, the critical surface mass
        unit_mass = 1e12
        sigma_crit = 2.35e09

        mass_map = input_map * unit_mass / sigma_crit
        average_mass_map = np.average(mass_map)
        self.fluctuation_field_map = mass_map - average_mass_map

    def calculate_field_size(self):
        deg2arcsec = 60.0*60.0
        deg2rad = 2*np.pi/360.0
        sqdeg2ster = deg2rad**2

        # pixel size in degrees
        self.pixel_size = np.abs(self.header['CDELT2'])

        self.shape = self.fluctuation_field_map.shape
        self.side_length_pixels = self.fluctuation_field_map.shape[0]

        self.side_length_deg = self.pixel_size*self.side_length_pixels
        self.side_length_rad = self.side_length_deg * deg2rad
        self.side_length_arcsec = self.side_length_deg * deg2arcsec

        self.fov_ster = self.side_length_rad**2


    def get_grid(self):
        tet_1grid_old = np.append((0.1+10**(.065*np.arange(50))), 1533.0)
        tet_1grid_new=np.zeros(24,dtype=float)
        tet_1grid_new[0]=3.41131121e+00
        tet_1grid_new[1]=7.09841996e+00
        tet_1grid_new[2]=1.10647820e+01
        tet_1grid_new[3]=1.48910839e+01
        tet_1grid_new[4]=1.72790839e+01
        tet_1grid_new[5:22]=tet_1grid_old[20:37]
        tet_1grid_new[22]=3.98207171e+02
        tet_1grid_new[23]=1.53300000e+03
        tet_1grid=np.zeros(24,dtype=float)
        tet_1grid=tet_1grid_new

        return tet_1grid

    def compute_power_spectrum(self):
        """
        Compute the angular power spectrum of fluctuation_field_map
        """

        # take the Fourier transform of the input map:
        fourier_amplitude_map = fftengine.fft2(self.fluctuation_field_map)/self.fluctuation_field_map.size
        fourier_amplitude_map = fftengine.fftshift(fourier_amplitude_map)
        fourier_power_spectrum = np.abs(fourier_amplitude_map)**2

        # Fourier space bins in arcsec^-1
        grid = self.get_grid()
        kbins = 1.0 / grid
        kbins = kbins[::-1]

        # binning
        scale = 1.0/self.side_length_arcsec
        y, x = np.indices(self.shape) * scale

        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

        r = np.hypot(x - center[0], y - center[1])
        #print(r)
        kbin_centers = 0.5*(kbins[1:]+kbins[:-1])
        #print(kbin_centers)
        #print(1.0/r[500])
        #print(1.0/kbin_centers)
        nr = np.histogram(r.flatten(), kbins)[0]

        power = np.histogram(r.flatten(), kbins, weights=fourier_power_spectrum.flatten())[0] / nr
        power *= self.fov_ster

        return (nr, kbin_centers, power)
