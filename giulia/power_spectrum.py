#PROGRAM FOR THE COMPUTATION OF POWER POWER SPECTRUM
#Python version:python3

import numpy as np
from numpy import fft
import math

arcsec2rad = math.pi/(180.0*3600.0)
ster2sqdeg = 3282.8
ster2sqarcmin = ster2sqdeg * 3600.0
ster2sqarcsec = ster2sqdeg * 3600.0 * 3600.0

def calc_f_clip (data_mask) :

    '''
    calculate unmasked sky fraction
    '''

    flat_cmask = data_mask.flatten()
    hn0 = flat_cmask [ np.where(flat_cmask > 0)]

    f_clip = float(hn0.size)/float(flat_cmask.size)
    print ("fraction of unmasked area =", f_clip)

    return f_clip

def azimuthal_average ( image, bins, scale ) :
    """
    image: power spectrum, abs(amp)**2
    bins: space bins in arcsec^-1 increasing (1/tet_1grid)
    scale: 1/side length in arcsec; 1.0/(pixel*nx_tot)
    """
    y, x = np.indices(image.shape)*float(scale)
    print ("y,x =",y,x)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    print ("center",center)
    r = np.hypot(x - center[0], y - center[1])
    print ("r",r)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    print ("bin_centers",bin_centers)
    nr = np.histogram(r,bins)[0]

    radial_prof = np.histogram(r, bins, weights=image)[0] / np.histogram(r, bins)[0]

    return nr, bin_centers, radial_prof

def auto_power_obs (data_obj, data_mask, tet_1grid, pixel, outfile = None, writefits = None ) :
    """
    data_obj: fluctuation field (data_mass*ecf_mass/sigma_crit - average_mass)
    tet_1grid: bin edges in arcsecs, (3, 1533)
    pixel: pixel size in arc
    """
    nx_tot = data_obj.shape[0]
    ny_tot = data_obj.shape[1]
    nx21 = nx_tot/2+1
    ny21 = ny_tot/2+1

    # full FOV in steradians; pixel in arcsecs
    area = float(nx_tot)*float(ny_tot)*(pixel/3600.0)**2.0/ster2sqdeg
    print ("FOV in steradian =", area)

    mask = (data_mask > 0)

    f_clip = calc_f_clip ( data_mask )

    weight = np.zeros([nx_tot,ny_tot])
    weight = data_mask

    # Compute fft;
    # Scipy FFT does not normalized result with 1/N, unlike IDL
    del_flux_fft = fft.fft2(data_obj)/data_obj.size 

    amp = del_flux_fft
    amp = fft.fftshift(del_flux_fft)

    # Auto power spectrum
    ps2d = np.abs(amp)**2

    # Fourier space bins in arcsec^-1; tet_1grid in arcsecs
    k_minmax = 1.0 /tet_1grid
    print ("k_minmax after binning =",k_minmax)
    # make bins monotonically increasing for azimuthal_averaging
    k_minmax = k_minmax[::-1]
    print ("monotonically increasing k_minmax =",k_minmax)

    # pixel in arcsec
    pairs, bin_center, power = azimuthal_average ( ps2d, k_minmax, 1.0/(pixel*nx_tot)  )

    power *= area/f_clip

    sig_p = power/(np.sqrt(0.5*pairs))

    return (bin_center, pairs, amp, power, sig_p)

def cross_power_abs(data_im,data_m,ampl1,ampl2,ampl3,nq1,nq2,nq3,auto1,auto2,auto3,tet_1grid,pixel):
 
    power_spec_a=np.real(ampl1)*np.real(ampl3)+np.imag(ampl1)*np.imag(ampl3)
    power_spec_b=np.real(ampl2)*np.real(ampl3)+np.imag(ampl2)*np.imag(ampl3)

    nx_tot = data_im.shape[0]
    ny_tot = data_im.shape[1]

    # full FOV in steradians; pixel in arcsecs
    area_im = float(nx_tot)*float(ny_tot)*(pixel/3600.0)**2.0/ster2sqdeg
    #print ("FOV in steradian =", area)

    mask = (data_m > 0)

    f_clip = calc_f_clip ( data_m )

    # Fourier space bins in arcsec^-1; tet_1grid in arcsecs
    k_minmax = 1.0 /tet_1grid
    #print ("k_minmax after binning =",k_minmax)
    # make bins monotonically increasing for azimuthal_averaging
    k_minmax = k_minmax[::-1]
    #print ("monotonically increasing k_minmax =",k_minmax)

    # pixel in arcsec
    pairs_a, bin_center_a, power_a = azimuthal_average ( power_spec_a, k_minmax, 1.0/(pixel*nx_tot)  )
    power_a *= area_im/f_clip

    pairs_b, bin_center_b, power_b = azimuthal_average ( power_spec_b, k_minmax, 1.0/(pixel*nx_tot)  )
    power_b *= area_im/f_clip

    sig_p_a = np.sqrt((auto1*auto3)/(nq1+nq3))
    sig_p_b = np.sqrt((auto2*auto3)/(nq2+nq3))

    cross_power = power_a-power_b
    sig_cross = np.sqrt(sig_p_a**2.0 + sig_p_b**2.0)    
    #sig_p_a = power/(np.sqrt(0.5*pairs))

    return (power_a,sig_p_a,power_b,sig_p_b,cross_power,sig_cross)

