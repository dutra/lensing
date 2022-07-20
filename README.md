# lensing

### Mennegheti
Meneghetti's power spectrum produces results inunits $l^2 P(l)$ vs $l$
fft.fftfreq result (wave vector) result is from 0 to 0.5 (Nyquist rate), then scaled to 0 to map length in pixels

## Data
### abell2744_mass_density.fits
Abell 2744 surface mass density map (in units of 10^12 Msun)
- Dimensions in pixels: 1000 x 1000
- Dimensions: 0.27805583361 or 1001 arcsec
- Conversion (pixel to deg): 0.000278055833611389
- Conversion (pixel to arcsec): 1.001
- Distance z=0.308
- Kpc/arcsec at z=0.308: 4.535624
- 1 Kpc = 3261.56 lyrs
- 14793.22 ly / arcsec

## Units
### Giulia's
Giulia's angular space bin edges `tet_1grid` is given in arcsecs.
Fourier space square units are in 1/arsec, scaled by 1/width in arcsecs.
Fourier space k units are 1/arcsec, scaled by 1/radius in arcsecs.

Fourier k bin center values are the average of the bin edges `0.5*(bins[1:]+bins[:-1])`.

The angular values `theta_mass` are given by 1/k_values, units in arcsecs.


The power is scaled by total squared area in steradians.


#### Mass auto power spectrum
Graph q^2 P(q)/2pi vs 2pi/q
2pi/q: theta (arcsec)
q: 2pi/theta (rad)
q^2 P(q)/2pi (rad)

### Meneghetti's

Fourier space in width pixels
kvals in width pixels * 2pi / fieldsize in rad
Power times (fielsize in rad / 2pi)^2
