import math
import numpy as np
from numpy import fft
from numpy import pi
from matplotlib import pyplot as plt
import scipy.stats as stats
import astropy.io.fits as pyfits
from scipy import ndimage
import pylab

from utils import Units


class Lens(object):
    def __init__(self, input_map):
        self.input_map_filepath = input_map

        self.load_file()
        self.calculate_field_size()

        self.units = Units()

    def load_file(self):
        # read convergence map from file:
        input_map, self.header = pyfits.getdata(self.input_map_filepath, header=True)
        # The 'meanmass.fits' is a surface density map (M_sun/kpc^2)
        # in units of 10^12 M_sun
        # To make it dimensionless divide by Sigma_crit=(c^2/4piG)*D_os*D_ol/D_ls, the critical surface mass
        unit_mass = 1e12
        sigma_crit = 2.35e09

        self.mass_map = input_map * unit_mass / sigma_crit
        average_mass_map = np.average(self.mass_map)
        self.fluctuation_field_map = (self.mass_map - average_mass_map)/average_mass_map


    def preview_mass_map(self):
        fig,ax = plt.subplots(1,1,figsize=(20,20))
        plt.imshow(self.mass_map, cmap='cividis')
        plt.colorbar()
        plt.show()

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


    def print_info(self):

        print(f"Shape: {self.shape}")
        print(f"Arsecs per pixel: {self.pixel_size*3600}")
        print(f"Side length: {self.side_length_arcsec} (arcsec)")
        r = np.hypot(self.side_length_arcsec/2.0, self.side_length_arcsec/2.0)
        print(f"Max angular radius: {r}")
        print(f"FOV: {self.fov_ster} (steradians)")

    def print_header(self):
        print("------------- HEADER ------------")
        for k, v in self.header.items():
            print(f"{k}: {v}")

        print("---------------------------------")


    def get_kbins_uniform(self, data, n=100):
        interval = (0, np.max(data))
        rbins = np.linspace(*interval, n+1)
        return rbins

    def get_kbins_uniform_ang(self, data, n=100):
        angular_bins = np.linspace(1.0, 1533, n+1)
        print(angular_bins)
        kbins = 1.0 / angular_bins
        kbins = kbins[::-1]

        scale = 1/self.side_length_arcsec
        return kbins/scale


    def get_kbins_equal_frequency(self, data, n=100):
        interval = (0, data.size)
        rbins = np.interp(np.linspace(0, data.size, n+1),
                          np.arange(data.size),
                          np.sort(data))

        return rbins

    def get_kbins_experimental(self, data, n=100):

        R = np.hypot(self.side_length_arcsec/2.0, self.side_length_arcsec/2.0)
        print(R)
        thetaa = np.array([R, R/3.0])
        kbins = np.zeros(3)
        kbins[0] = 0.0
        kbins[1] = 2.0/thetaa[0] - kbins[0]
        kbins[2] = 2.0/thetaa[1] - kbins[1]

        kbins *= self.side_length_arcsec
        print(kbins)

        return kbins


    def get_kbins_giulia(self):
        # bin edges in arcsecs
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

    def get_kbins_giulia_cats(self):
        # bin edges in arcsecs
        tet_1grid_old = np.append((0.1+10**(.065*np.arange(50))), 1533.0)
        tet_1grid_new=np.zeros(24,dtype=float)
        tet_1grid_new[0]=3.41131121e+00
        tet_1grid_new[1]=7.09841996e+00
        tet_1grid_new[2]=1.10647820e+01
        tet_1grid_new[3]=1.48910839e+01
        tet_1grid_new[4]=1.72790839e+01
        tet_1grid_new[5:22]=tet_1grid_old[20:37]
        tet_1grid_new[22]=3.98207171e+02
        tet_1grid_new[23]=8.0000000e+02
        tet_1grid=np.zeros(24,dtype=float)
        tet_1grid=tet_1grid_new

        grid = tet_1grid
        print(grid)

        kbins = 1.0 / grid
        kbins = kbins[::-1]

        scale = 1/self.side_length_arcsec
        return kbins/scale

    def get_kbins(self, method, data=None):
        if method == "giulia":
            return self.get_kbins_giulia()
        if method == "giulia_cats":
            return self.get_kbins_giulia_cats()
        if method == "equal_frequency":
            return self.get_kbins_equal_frequency(data)
        if method == "experimental":
            return self.get_kbins_experimental(data)
        if method == "uniform_k":
            return self.get_kbins_uniform(data)
        if method == "uniform_angle":
            return self.get_kbins_uniform_ang(data)

    def set_kbins(self, kbins):
        self.kbins = kbins

    def compute_power_spectrum(self, kbin_method="giulia"):
        """
        Compute the angular power spectrum of input_map
        :param input_map: input map (n x n numpy array)
        :param field_size: the side-length of the input map in degrees
        :return: l, Pl - the power-spectrum at l
        """

        # take the Fourier transform of the input map:
        fourier_map = fft.fftn(self.fluctuation_field_map)/np.prod(self.fluctuation_field_map.shape)
        fourier_map = fft.fftshift(fourier_map)


        # compute the Fourier amplitudes
        fourier_amplitudes = np.abs(fourier_map)**2
        fourier_amplitudes = fourier_amplitudes.flatten()

        # compute the wave vectors
        # returns from 0 to 0.5 (Nyquist rate)
        kfreq = fft.fftfreq(self.fluctuation_field_map.shape[0])*self.fluctuation_field_map.shape[0]
        kfreq = fft.fftshift(kfreq)
        kfreq2D = np.meshgrid(kfreq, kfreq)

        # take the norm of the wave vectors
        kfreq_norm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        kfreq_norm = kfreq_norm.flatten()

        # set up k bins. The power spectrum will be evaluated in these bins
        # kbins = self.get_kbins_equal_frequency(kfreq_norm, 100)
        #kbins = self.get_kbins_uniform_ang(kfreq_norm, 100)
        # kbins = self.get_kbins_giulia()
        kbins = self.get_kbins(kbin_method, kfreq_norm)
        kbins[0] = 0
        print(kbins/self.side_length_arcsec)

        # use the middle points in each bin to define the values of k
        # where the PS is evaluated
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        nk = np.histogram(kfreq_norm, kbins)[0]
        # now compute the PS: calculate the mean of the
        # Fourier amplitudes in each kbin
        # Pbins = np.histogram(kfreq_norm, rbins, weights=fourier_amplitudes.flatten())[0] / nr
        Pbins, _, _ = stats.binned_statistic(kfreq_norm, fourier_amplitudes,
                                             statistic="mean", bins=kbins)


        # set the unit conversion factor
        # k = 2pi/wavelength
        # factor scales from 0 to side length in rads to 0 to 2 pi
        #scale = 2.0*np.pi/self.field_size
        scale = 1/self.side_length_arcsec

        kvals = kvals*scale
        Pl = Pbins*self.fov_ster

        self.nr = nk
        self.kbin_centers = kvals
        self.power = Pl

        return nk, kvals, Pl


    def make_bin_info_plot(self):

        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        theta_bin_centers = 1.0/kbin_centers

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.set_yscale('log')
        ax.plot(theta_bin_centers, nr, 'o')
        ax.set_xlabel(r'$\theta$ (arcsec)')
        ax.set_ylabel(r'#r per bin')
        ax.set_ylim(1, np.max(nr)*1.2)
        plt.tight_layout()
        #plt.savefig('abell2744_power_theta.jpeg')
        pylab.show()


    def print_bin_info(self):
        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        theta_bin_centers = 1.0/kbin_centers

        for x, n in zip(np.flip(theta_bin_centers), np.flip(nr)):
            print(f"{x:0.4f}: {n}", end='\t')

    def make_power_spectrum_linear_bar_plot(self, outfile=None):
        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        sig_power = power/(np.sqrt(0.5*nr))

        theta_bin_centers = 1.0/kbin_centers
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.errorbar(theta_bin_centers, power, yerr=sig_power, color='C0', fmt='o')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$2\pi/q$ [arcsec]')
        ax.set_ylabel(r'$P_\kappa(q) $')
        ax.title.set_text('Mass Auto-Power Spectrum')
        plt.tight_layout()
        #plt.savefig('output/abell2744_power_meneghetti.jpeg')
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


    def make_power_spectrum_plot(self, outfile=None):
        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        theta_bin_centers = 1.0/kbin_centers

        power_smooth = ndimage.gaussian_filter1d(power, sigma=0.8)
        sig_power = power/(np.sqrt(0.5*nr))
        y_plus = power_smooth + sig_power
        y_minus = power_smooth - sig_power

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(theta_bin_centers, power_smooth, color='C0')
        ax.fill_between(theta_bin_centers, y_plus, y_minus, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$2\pi/q$ [arcsec]')
        ax.set_ylabel(r'$P_\kappa(q) $')

        #ax.set_xlim(3.602922101421851, 808.5723924286702)
        #ax.set_ylim(0.09120854396955308, 4.817770706717263)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()

    def make_power_spectrum_q2_plot(self, smooth=True, outfile=None):

        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        theta_bin_centers = 1.0/kbin_centers

        qbin_centers = 2.0*np.pi*kbin_centers
        qbin_centers = qbin_centers/self.units.arcsec2rad

        sig_power = power/(np.sqrt(0.5*nr))
        sig_q_power = sig_power*qbin_centers**2/(2.0*math.pi)

        power_q = power*qbin_centers**2/(2.0*math.pi)

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        if smooth:
            power_q_smooth = ndimage.gaussian_filter1d(power_q, sigma=0.8)
            ax.title.set_text('mass auto-power spectrum (smoothed)')
        else:
            power_q_smooth = power_q
            ax.title.set_text('mass auto-power spectrum (not smoothed)')
        y_plus = power_q_smooth + sig_q_power
        y_minus = power_q_smooth - sig_q_power


        ax.plot(theta_bin_centers, power_q_smooth, 'xr')
        ax.plot(theta_bin_centers, power_q_smooth, color='C0')
        ax.fill_between(theta_bin_centers, y_plus, y_minus, alpha=0.3)
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$2\pi/q$ [arcsec]')
        ax.set_ylabel(r'$q^2 P_\kappa(q) / 2\pi$')
        #ax.set_xlim(3.602922101421851, 808.5723924286702)
        #ax.set_ylim(0.09120854396955308, 4.817770706717263)
        # secax = ax.secondary_xaxis('top', functions=(arcsec2dist,dist2arcsec))
        # secax.set_xlabel('distance [kpc]')
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()

    def make_bin_distribution_plot(self, outfile=None):
        nr = self.nr
        kbin_centers = self.kbin_centers
        power = self.power

        theta_bin_centers = 1.0/kbin_centers
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(theta_bin_centers, nr, 'o')
        ax.set_xlabel(r'k bin centers')
        ax.set_ylabel(r'#r per bin')
        ax.set_ylim(0, np.max(nr)*1.2)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()
