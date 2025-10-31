import numpy as np
from constant import *

class Matter:
    def __init__(self):
        pass

    def _x(self, rhob: float) -> float:
        """Dimensionless degeneracy parameter x = p_F / (m c)"""
        return rhob ** (1/3)
    
    def gamma(self, rhob):
        x = self._x(rhob)
        return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    

class DarkMatter(Matter):
    """Equation of state for a single component"""
    def __init__(self, 
                 m, # mass of the dark matter [MeV]
                 k=None, # decay constant [s-1]
                 dE=None, # Gamma ray emitted by the decaying dark matter [MeV]
                 Z=None # atmoic number of the white dwarf
                 ):
        self.m = m * 1e6 * C_e # mass of the dark matter particle [g]
        super().__init__()

        if k is not None and dE is not None:
            self.k = k
            self.dE = dE * 1e9 * C_e # [GeV] to [erg]
            self.Z = Z
            self.decay = True
        else:
            self.decay = False
    
    
    def pressure(self, rhob: float) -> float:
        """Pressure in code units (energy density)"""
        x = self._x(rhob)
    
        # Fermi degeneracy pressure (same form, normalized by DM mass)
        P = (x * np.sqrt(x**2 + 1) * (2*x**2 - 3) + 
                3 * np.arctanh(x / np.sqrt(x**2 + 1))) / 8
    
        return P
    
    def gr_sigma(self):
        # Calculating cross section of Gamma ray
        # Using Fornalski Equation
        # E_gamma: gamma ray photon energy in erg
        sigma = 0
        E_ratio = self.dE / (E_e * C_e)
        for i in range(6):
            sum = 0
            for j in range(4):
                sum += a_ij[j][i] * (self.Z ** j)

            sigma += (np.log(E_ratio) ** i) * sum
        return sigma * 1e-24
    
    def luminosity(self, m):
        """
        m: enclosed mass [g]
        """

        L = self.k * self.dE * m / self.m # unit: erg/s
        return L

    def photon_pressure(self, r, m, n):
        # return dpdr by due to the decay dm
        """
        r: radius [cm]
        m: enclosed mass [g]
        n: number density of the nucleon
        """
        if not self.decay:
            raise ValueError("This is not a decaying dark matter. STOP")
        
        sigma = self.gr_sigma() 
        L = self.luminosity(m) # erg /s
        l = 1 / (sigma * n) # unit = cm

        dudr = 3 * L / (4 * np.pi * r ** 2 * l * c) # erg cm^-4 s^-1
        return -dudr/3
    
class NormalMatter(Matter):

    def __init__(self, Ye):
        #m: Ye for nm
        self.Ye = Ye
        super().__init__()

    def pressure(self, rhob: float) -> float:
        """Pressure in code units (energy density)"""
        x = self._x(rhob)
        
        P = self.Ye * me * (x * np.sqrt(x**2 + 1) * (2*x**2 - 3) + 
                        3 * np.arctanh(x / np.sqrt(x**2 + 1))) / (8 * mp)
        
        return P
    