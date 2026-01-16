"""
CGS Unit is used.
"""

import numpy as np

# General Constant
me = 9.1093837e-31 * 1000 # g
mp = 1.67262192e-27 * 1000 # g
c = 299792458 * 100 # cm/s
G = 6.67430e-8 # cgs
C_e = 1.60218e-12 # ***Coulomb statCoulomb, use to eV->erg
statC_e = 4.803e-10
E_e = me * (c ** 2) / C_e # m_e c^2 in eV
r_e =  2.8179403205e-15 * 100 # electron radius in cm
r_p = 0.877e-15 * 100 # proton radius in cm
proton_energy = 938.272e6 * C_e  # erg
M_SOLAR = 1.98e33 # g
L_SOLAR = 3.828e33 # erg/s
h = 6.626e-27 # planck constant in cgs
hbar = h / (2 * np.pi)
mH = 1.6735e-24 # mass of hydrogen in g
kB = 1.38e-16 # boltzmann constant in cgs
sigma = 5.67e-5 # stefann constant
a = 4 * sigma / (3 * c)