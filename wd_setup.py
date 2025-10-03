import numpy as np
from constants import *
from decay import decay

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, Z, k):
        self.Ye = Ye
        self.rhoc_scaled = rhoc_scaled
        self.p_decay = decay(k=k, Z=Z, Ye=Ye)

        # scale
        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.n0 = 5.89e29

    def _gamma(self, x):
        return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    
    def _x(self, rhob):
        return (rhob) ** (1/3)

    def get_derivative(self, state, rb): 
        # state = [rho, m]
        rho, m = np.maximum(state[0], np.finfo(np.float32).eps), np.maximum(state[1], np.finfo(np.float32).eps)

        x = self._x(rho)
        gamma = self._gamma(x)

        P = self.Ye * me * (x * np.sqrt(x ** 2 + 1) * (2 * x ** 2 -3) + 3 * np.arctanh(x / np.sqrt(x ** 2 + 1))) / (8 * mp)

        dmdr = rb**2 * rho 

        if rb < 1e-6:
            #m = (1/3) * rho * (rb**3)
            t0 = rb * rho ** 2 / 3
            t1 = (1 + P/rho)
            t2 = (1 + P/ (rho /3))
            t3 = 1 / (1 - self.Ye * me * (2/3) * rho * rb ** 2 / mp)
        else:
            t0 = (m * rho / rb**2)
            t1 = (1 + P/rho)
            t2 = (1 + (rb**3 * P)/m)
            t3 = 1 / (1 - (2 * m * self.Ye * me / mp)/rb)

        # consider effect of proton decay
        n_p = self.Ye * rho * self.rho0 / mp # proton density, unit = cm-^3, Yp=Ye
        dPdr = -t0 * t1 * t2 * t3 - self.p_decay.photon_pressure(E_gamma=pion_photon, r=rb*self.R0, m=m*self.M0, n=n_p) * self.R0 / self.rho0 * 2 - - self.p_decay.photon_pressure(E_gamma=positron_photon, r=rb*self.R0, m=m*self.M0, n=n_p) * self.R0 / self.rho0 * 2
        #print(-t0 * t1 * t2 * t3, - self.p_decay.photon_pressure(E_gamma=pion_photon, r=rb*self.R0, m=m*self.M0, n=n_p) * self.R0 / self.rho0, dPdr / gamma, self.p_decay.drhodr(n_p=n_p) * self.R0 / self.rho0)
        drhodr = dPdr / gamma - self.p_decay.drhodr(n_p=n_p) * self.R0 / self.rho0

        return np.array([drhodr, dmdr]), [t1, t2, t3, P]