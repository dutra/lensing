import numpy as np
from numpy import fft as fft


class Binning(object):
    def __init__(self, scale, method='experimental', **kargs):
        self.scale = scale
        self.method = method

        print(kargs)
        match method:
          case 'experimental':
            self.kbins = self.build_kbins_experimental(**kargs)
          case 'giulia':
            self.kbins = self.get_kbins_giulia(**kargs)
          case 'equal_frequency':
            self.kbins = self.get_kbins_equal_frequency(**kargs)
          case 'slicer':
            self.kbins = self.get_kbins_slicer(**kargs)
          case _:
            print("Method: {method}")

    def get_method(self):
        return self.method

    def get_kbins(self):
        return self.kbins

    def get_kbins_slicer(self, side_length=1000, min1=100, min2=10000):
        kfreq = fft.fftfreq(side_length)*side_length
        kfreq = fft.fftshift(kfreq)
        kfreq2D = np.meshgrid(kfreq, kfreq)

        # take the norm of the wave vectors
        kfreq_norm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        kfreq_norm = kfreq_norm.flatten()

        x = np.arange(10)*min1
        while x[-1] < kfreq_norm.size-min2:
            x = np.append(x, x[-1]+min2)

        rbins = np.interp(x,
                        np.arange(kfreq_norm.size),
                        np.sort(kfreq_norm))

        return rbins

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
        return kbins/self.scale


    def get_kbins_equal_frequency(self, data=[], n=100):
        interval = (0, data.size)
        rbins = np.interp(np.linspace(0, data.size, n+1),
                          np.arange(data.size),
                          np.sort(data))

        return rbins

    def build_kbins_experimental(self, R=None, n=None, m=None):
        b = np.log10(R)/m
        print(f"{b=}")
        c = 10**b
        print(f"{c=}")

        kbins = [0, 2.0/R]
        kbins.append(kbins[-1]*c*c)

        for i in range(n-2):
            kbins.append(kbins[-1]*c)

        return np.array(kbins)*self.scale


    def check_kbins(kbins):
        for i, k in enumerate(kbins):
            for j, l in enumerate(kbins[:i]):
                if k < l:
                    return (i, j)
        return None

    def build_theta_vals(kbins):
        kbins /= self.scale
        kvals = 0.5*(kbins[1:]+kbins[:-1])
        theta_vals = 1.0/kvals
        theta_vals = np.flip(theta_vals)
        return theta_vals

    #def nr_histogram(kbins, ):


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

        return kbins*self.scale

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
