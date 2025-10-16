"""
CGS Unit is used.
"""

import numpy as np

# General Constant
me = 9.1093837e-31 * 1000 # g
mp = 1.67262192e-27 * 1000 # g
c = 299792458 * 100 # cm/s
G = 6.67430e-8 # cgs
C_e = 1.60217663e-12 # charge in e- in erg/ev
E_e = me * (c ** 2) / C_e # m_e c^2 in eV
r_e =  2.8179403205e-15 * 100 # electron radius in cm
r_p = 0.877e-15 * 100 # proton radius in cm
pion_photon = 135e6 / 2 * C_e # erg
positron_photon = 0.511e6 * C_e # erg
proton_energy = 938.272e6 * C_e  # erg
M_SOLAR = 1.98e33 # g

# Coefficients in Calculating gamma ray cross section
row_0 = [
    0.083089, -0.08717743, 0.02610534, 
    -2.74655e-3, 4.39504e-5, 9.05605e-6, 
    -3.97621e-7
]

row_1 = [
    0.265283, -0.10167009, 0.00701793, 
    2.371288e-3, -5.020251e-4, 3.6531e-5, 
    -9.4644e-7
]

row_2 = [
    2.18838e-3, -2.914205e-3, 1.26639e-3, 
    -7.6598e-5, -1.58882e-5, 2.18716e-6, 
    -7.49728e-8
]

row_3 = [
    -4.48746e-5, 4.75329e-5, -1.43471e-5, 
    1.19661e-6, 5.7891e-8, -1.2617e-8, 
    4.633e-10
]

row_4 = [
    6.29882e-7, -6.72311e-7, 2.61963e-7, 
    -5.1862e-8, 5.692e-9, -3.29e-10, 
    7.7e-12
]

a_ij = np.array([row_0, row_1, row_2, row_3, row_4], dtype=np.float64)