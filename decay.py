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
        # E_gamma: gamma ray photon energy in erg
        sigma = 0
        E_ratio = E_gamma / (E_e * C_e)
        for i in range(6):
            sum = 0
            for j in range(4):
                sum += a_ij[j][i] * (self.Z ** j)

            sigma += (np.log(E_ratio) ** i) * sum
        return sigma * 1e-24
    
    def luminosity(self, E_gamma, m):
        """
        m: enclosed mass in g
        k: decay constant in 1/s
        E_gamma: gamma ray energy in erg
        mp: proton mass in g
        """
        return self.Ye * m * self.k * E_gamma/ mp # unit: erg/s

    def photon_pressure(self, E_gamma, r, m, n):
        # n = np = ne, but with dimension
        # but Ye = Yp since no of proton = electron in atom
        # return dpdr by species
        
        sigma = self.gr_sigma(E_gamma=E_gamma) 
        L = self.luminosity(E_gamma, m) # erg /s
        l = 1 / (sigma * n) # unit = cm

        dudr = 3 * L / (4 * np.pi * r ** 2 * l * c) # erg cm^-4 s^-1
        return -dudr/3
    
    def drhodr(self, n_p):
        # positron annhilation doesn't change the density
        # therefore, only the effect of losing proton by proton decay is considered
        # np here is not dimensionless

        return self.k * n_p * mp / self.Ye # g cm^-3 s^-1