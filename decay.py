import numpy as np
from constants import *

class decay:
    def __init__(self, k, Z, Ye):
        self.k = k
        self.Z = Z
        self.Ye = Ye

    def gr_sigma(self, E_gamma):
        # Calculating cross section of Gamma ray
        # Using Fornalski Equation
        # E_gamma: gamma ray photon energy in eV
        sigma = 0

        E_ratio = E_gamma / E_e 
        
        for i in range(6):
            sum = 0
            for j in range(4):
                sum += a_ij[j][i] * (self.Z ** j)

            sigma += (np.log(E_ratio) ** i) * sum

        return sigma
    
    def luminosity(self, E_gamma, m):
        # E_gamma in j
        return 4 * np.pi * self.Ye * m * self.k * E_gamma/ mp # unit: erg/s

    def _photon_pressure(self, E_gamma, r, m, n, r_spec):
        # n = np = ne, but with dimension
        # but Ye = Yp since no of proton = electron in atom
        # return dpdr by species
        # r_spec is radius of proton or electron depends on the input
        
        sigma = np.pi * r_spec ** 2
        L = self.luminosity(E_gamma, m) # erg /s
        l = 1 / (np.sqrt(2) * sigma * n) # unit = cm

        dudr = 3 * L / (4 * np.pi * r ** 2 * l * c) # erg cm^-4 s^-1

        return dudr / 3
    
    def photon_pressure(self, E_gamma, r, m, n):
        return self._photon_pressure(E_gamma, r, m, n, r_p) + self._photon_pressure(E_gamma, r, m, n, r_e)
    
    def drhodr(self, n_p):
        # positron annhilation doesn't change the density
        # therefore, only the effect of losing proton by proton decay is considered
        # np here is not dimensionless

        return self.k * n_p * mp / self.Ye # g cm^-3 s^-1