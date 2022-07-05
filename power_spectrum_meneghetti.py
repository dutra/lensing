import math
import numpy as np
from numpy import fft as fftengine
from numpy import pi
import scipy.stats as stats
import astropy.io.fits as pyfits
import pylab

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

    def get_kbins(self):
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

        grid = tet_1grid

        kbins = 1.0 / grid
        kbins = kbins[::-1]

        scale = 1/self.side_length_arcsec
        return kbins/scale

    #def get_kbins(self):

    def compute_power_spectrum(self):
        """
        Compute the angular power spectrum of input_map
        :param input_map: input map (n x n numpy array)
        :param field_size: the side-length of the input map in degrees
        :return: l, Pl - the power-spectrum at l
        """

        # take the Fourier transform of the input map:
        fourier_map = fftengine.fftn(self.fluctuation_field_map)/np.prod(self.fluctuation_field_map.shape)

        # compute the Fourier amplitudes
        fourier_amplitudes = np.abs(fourier_map)**2
        fourier_amplitudes = fourier_amplitudes.flatten()

        # compute the wave vectors
        # returns from 0 to 0.5 (Nyquist rate)
        kfreq = fftengine.fftfreq(self.fluctuation_field_map.shape[0])*self.fluctuation_field_map.shape[0]
        kfreq2D = np.meshgrid(kfreq, kfreq)

        # take the norm of the wave vectors
        kfreq_norm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        kfreq_norm = kfreq_norm.flatten()

        # set up k bins. The power spectrum will be evaluated in these bins
        half_width = self.side_length_pixels/2
        # calculate radius of corner
        R = math.ceil(np.sqrt(half_width**2 + half_width**2))
        # create bins from 0 to R in size 1 increment
        #rbins = np.linspace(0.0, R, (R+1))
        rbins = self.get_kbins()

        # set the unit conversion factor
        # k = 2pi/wavelength
        # factor scales from 0 to side length in rads to 0 to 2 pi
        #scale = 2.0*np.pi/self.field_size
        scale = 1/self.side_length_arcsec

        # use the middle points in each bin to define the values of k
        # where the PS is evaluated
        kvals = 0.5 * (rbins[1:] + rbins[:-1])

        nr = np.histogram(kfreq_norm, rbins)[0]
        # now compute the PS: calculate the mean of the
        # Fourier amplitudes in each kbin
        # Pbins = np.histogram(kfreq_norm, rbins, weights=fourier_amplitudes.flatten())[0] / nr
        Pbins, _, _ = stats.binned_statistic(kfreq_norm, fourier_amplitudes,
                                            statistic="mean", bins=rbins)

        # return kvals and PS
        l = kvals*scale
        Pl = Pbins*self.fov_ster

        return nr, l, Pl
