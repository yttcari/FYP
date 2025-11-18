import numpy as np

from constant import *
from rk4 import rk4
from matter import *

class WhiteDwarf:
    def __init__(self,
                 nm_rhoc, # core density of normal matter [g/cc]
                 Ye, # electron fraction
                 pdm_rhoc=None, # core density of parent dark matter [g/cc]
                 ddm_rhoc=None, # core density of daughter dark matter [g/cc]
                 k=None, # decay constant of parent dark matter [s-1]
                 Z=None, # atomic number of normal matter 
                 pdm_m=None, # mass of parent dark matter particle [GeV]
                 ddm_m=None, # mass of daughter dark matter particle [GeV]
                 ):
        self.Ye = Ye
        self.nm = NormalMatter(self.Ye)

        # scale
        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.n0 = 5.89e29

        # dimensionless variables declaration
        self.nm_rhoc = nm_rhoc / self.rho0

        if pdm_rhoc is not None and ddm_rhoc is not None and k is not None and Z is not None and pdm_m is not None and ddm_m is not None:
            assert pdm_m >= ddm_m # ensure parent dark matter has larger mass than daughter dark matter
            self.decay_dm = True
            self.pdm = DarkMatter(m=pdm_m, k=k, dE=(pdm_m-ddm_m), Z=Z)
            self.ddm = DarkMatter(m=ddm_m)
            self.pdm_rhoc = pdm_rhoc / self.rho0
            self.ddm_rhoc = ddm_rhoc / self.rho0
            self.k = k
            self.Z = Z
            
            self.pdm_m = pdm_m * 1e9 * C_e # [erg]
            self.ddm_m = ddm_m * 1e9 * C_e # [erg]
        else:
            self.decay_dm = False

    def TOV(self, rb, rho, m, P, total_P, total_m, total_rho):
        """
        rb: dimensionless radius
        rho: density of the species
        P: species pressure
        m: species mass
        total_P: degeneracy pressure of the species
        total_m: total enclosed mass
        """
        if rb <= 1e-6:
            total_m = (1/3) * total_rho * (rb**3)

        t0 = (total_m * rho / rb**2)
        t1 = (1 + P/max(rho, np.finfo(np.float32).eps))
        t2 = (1 + (rb**3 *total_P)/total_m)
        t3 = 1 / (1 - (2 * total_m * self.Ye * me / mp)/rb)

        return -t0 * t1 * t2 * t3

    def get_derivative(self, state, rb):
        if self.decay_dm:
            # extract state vector variable
            nm_rho = max(state[0], 0)
            pdm_rho = max(state[1], 0)
            ddm_rho = max(state[2], 0)

            nm_M = max(state[3], 0)
            pdm_M = max(state[4], 0)
            ddm_M = max(state[5], 0)

            nm_P = self.nm.pressure(nm_rho)
            pdm_P = self.pdm.pressure(pdm_rho)
            ddm_P = self.ddm.pressure(ddm_rho)

            total_M = nm_M + pdm_M + ddm_M
            total_rho = nm_rho + pdm_rho + ddm_rho
            total_P = nm_P + pdm_P + ddm_P

            nm_dmdr = rb**2 * nm_rho
            pdm_dmdr = rb**2 * pdm_rho
            ddm_dmdr = rb**2 * ddm_rho

            nm_TOV = self.TOV(rb, rho=nm_rho, P=nm_P, m=nm_M, total_m=total_M, total_P=total_P, total_rho=total_rho)
            pdm_TOV = self.TOV(rb, rho=pdm_rho, P=pdm_P, m=pdm_M, total_m=total_M, total_P=total_P, total_rho=total_rho)
            ddm_TOV = self.TOV(rb, rho=ddm_rho, P=ddm_P, m=ddm_M, total_m=total_M, total_P=total_P, total_rho=total_rho)

            nucleon_photon_pressure = self.pdm.photon_pressure(r=rb*self.R0, m=pdm_M*self.M0, ne=nm_rho*self.rho0*self.Ye/mp, nn=nm_rho*self.rho0/mp) * self.R0 / self.rho0

            nm_drhodr = (nm_TOV - nucleon_photon_pressure) / max(self.nm.gamma(nm_rho), np.finfo(np.float32).eps)
            pdm_drhodr = pdm_TOV / max(self.pdm.gamma(pdm_rho), np.finfo(np.float32).eps)
            ddm_drhodr = ddm_TOV / max(self.ddm.gamma(ddm_rho), np.finfo(np.float32).eps)

            return np.array([nm_drhodr, pdm_drhodr, ddm_drhodr, nm_dmdr, pdm_dmdr, ddm_dmdr])
        else:
            nm_rho = max(state[0], 0)
            nm_M = max(state[1], 0)
            nm_P = self.nm.pressure(nm_rho)
            nm_TOV = self.TOV(rb, rho=nm_rho, P=nm_P, m=nm_M, total_m=nm_M, total_P=nm_P, total_rho=nm_rho)
            nm_drhodr = nm_TOV / self.nm.gamma(nm_rho)
            nm_dmdr = rb**2 * nm_rho
            return np.array([nm_drhodr, nm_dmdr])

    def integrate(self, verbose=False):
        # initial radius
        rb = 1e-3
        
        if self.decay_dm:
            # initial state vector
            state = np.array([
                self.nm_rhoc, # dimensionless normal matter density
                self.pdm_rhoc, # dimensionless parent dark matter density
                self.ddm_rhoc, # dimensionless daughter dark matter density
                (1/3) * self.nm_rhoc * (rb ** 3), # dimensionless normal matter mass
                (1/3) * self.pdm_rhoc * (rb ** 3), # dimensionless parent dark matter mass
                (1/3) * self.ddm_rhoc * (rb ** 3), # dimensionless parent dark matter mass
            ])
        else:
            state = np.array([self.nm_rhoc, # dimensionless normal matter density
                     (1/3) * self.nm_rhoc * (rb ** 3), # dimensionless normal matter mass
                     ])
            
        history = []

        while state[0] > 1e-10:
            dr = 5e-4 * rb
            history.append(np.array([rb] + list(state)))
            print(state)
            if verbose:
                print(f"radius (km): {self.rbar2r(rb):.3e}, rho (g/cc): {self.rhobar2rho(state[0]):.3e}, mass [Msolar]: {self.mbar2m(state[3]):.3e}")
            state = rk4(self.get_derivative, dr=dr, rb=rb, state=state)

            rb += dr

        self.history = np.array(history)

    def rbar2r(self, rbar):
        return rbar * self.R0 / (100 * 1000)

    def rhobar2rho(self, rhobar):
        return rhobar * self.rho0
    
    def mbar2m(self, mbar):
        return mbar * self.M0 / M_SOLAR