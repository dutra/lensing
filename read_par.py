import re
import csv
import math


with open('data/hlsp_frontier_model_abell2744_cats_v3.1.par', 'r') as f:
    par_file = f.read()

# Extract potential sections
potentiels_m = re.findall(r'potentiel.*?end', par_file, flags=re.MULTILINE|re.DOTALL)
if not potentiels_m:
    raise Exception("Invalid par file")

potentiels = {}
for p in potentiels_m:
    m = re.search(r'^potentiel (?P<value>[\w\.]+)\n', p)
    potentiel = m.groupdict()['value']
    m = re.findall(r'\t(?P<key>[a-z_]+)[ ]+(?P<value>-?\d+\.?\d+)', p)
    potentiels[potentiel] = dict(m)


# Calculate mass (Eq. 5.122 Meneghetti or Eq. 10 Limousin 2005)
# https://projets.lam.fr/projects/lenstool/wiki/PoTential
for k, p in potentiels.items():
    r_cut = float(p['cut_radius']) # arcsec
    r_cut_kpc = float(p['cut_radius_kpc'])
    r_core = float(p['core_radius']) # arcsec
    r_core_kpc = float(p['core_radius_kpc'])
    v_disp = float(p['v_disp']) # km/s
    G = 4.30091e-6 # kpc M_sun^-1 km/s^2

    m_tot = math.pi*v_disp**2/G * r_cut_kpc**2/(r_cut_kpc+r_core_kpc) # M_sun
    p['m_tot'] = m_tot

    #if r_cut_kpc > 5 and r_cut_kpc < 50:
    if m_tot > 0.9e11 and m_tot < 1.1e11:
        print(f"Potentiel {k}: Mass {m_tot:e}, r_core {r_core_kpc}, r_cut: {r_cut_kpc}")
