import numpy as np
from constants import *

class decay:
    def __init__(self, k, Z, Ye):
        self.k = k # s-1
        self.Z = Z # atomic number
        self.Ye = Ye # electron fraction

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
        return sigma * 1e-24 # cm^2
    
    def luminosity(self, E_gamma, m):
        """
        m: enclosed mass in g
        k: decay constant in 1/s
        E_gamma: gamma ray energy in erg
        mp: proton mass in g
        """
        L = self.Ye * m * self.k * E_gamma/ mp # unit: erg/s
        return L

    def photon_pressure(self, E_gamma, r, m, ne, nn):
        """
        r: radius [cm]
        m: mass [g]
        ne: electron number fraction
        nn: nucleon number fraction
        E_gamma: gamma ray energy [erg]
        """
        # n = np = ne, but with dimension
        # but Ye = Yp since no of proton = electron in atom
        # return dpdr by species
        
        sigma_gr_e = self.gr_sigma(E_gamma=E_gamma)
        sigma_n = np.pi * (0.877 * 1e-13) ** 2
        nsigma = sigma_gr_e * ne + sigma_n * nn
        L = self.luminosity(E_gamma, m) # erg /s
        l = 1 / nsigma # unit = cm

        dudr = 3 * L / (4 * np.pi * r ** 2 * l * c) # erg cm^-4 s^-1
        return -dudr/3