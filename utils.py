import numpy as np
from astropy.cosmology import FlatLambdaCDM



class Units(object):
    def __init__(self):
        self.z = 0.308

        self.arcsec2rad = np.pi/(180.0*3600.0)

        # Calculates distance in parsecs
        cosmo_flat = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.7255)
        self.arcsec2kpc = cosmo_flat.kpc_proper_per_arcmin(self.z).value/60.0

    def arcsec2dist(self):
        return lambda t: self.arcsec2kpc*t

    def dist2arcsec(self):
        return lambda t: t/self.arcsec2kpc
