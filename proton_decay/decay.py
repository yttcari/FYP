import numpy as np
from constants import *

class proton_decay:
    def __init__(self, k):
        self.k = k # s-1
    
    def luminosity(self, m):
        """
        m: enclosed mass in g
        k: decay constant in 1/s
        E_gamma: gamma ray energy in erg
        mp: proton mass in g
        """
        L = m * self.k * proton_energy/ mp # unit: erg/s
        return L
    
class PBH:
    def __init__(self, pbhM, Tc):
        self.pbhM = pbhM
        self.Tc = Tc