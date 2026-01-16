import numpy as np
from constants import *

hbar = h / (2 * np.pi)

class proton_decay:
    def __init__(self, k):
        self.k = k # s-1
    
    def luminosity(self, r, m):
        """
        m: enclosed mass in g
        k: decay constant in 1/s
        E_gamma: gamma ray energy in erg
        mp: proton mass in g
        """
        L = m * self.k * proton_energy/ mp # unit: erg/s
        return L
    
class PBH:
    def __init__(self, pbhM):
        self.pbhM = pbhM

    def get_mass(self):
        # return mass for TOV calculation
        return self.pbhM
    
    def hawking_T(self):
        return hbar * c ** 3 / (8 * np.pi * G * self.pbhM * kB)

    def luminosity(self, r, m):
        # input: r in cgs unit

        P = sigma * self.hawking_T() ** 4
        L = P / (4 * np.pi * r ** 2)

        return L # erg/s

class No_source:
    def __init__(self):
        pass

    def get_mass(self):
        return 0
    
    def lumionsity(self, **kwargs):
        return 0