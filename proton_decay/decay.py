import numpy as np
from constants import *

class decay:
    def __init__(self, k, Z, Ye):
        self.k = k # s-1
        self.Z = Z # atomic number
        self.Ye = Ye # electron fraction
    
    def luminosity(self, E_gamma, m):
        """
        m: enclosed mass in g
        k: decay constant in 1/s
        E_gamma: gamma ray energy in erg
        mp: proton mass in g
        """
        L = m * self.k * E_gamma/ mp # unit: erg/s
        return L

    def deg_mean_free_path(self, T, rho):
        # ======= Assumed Carbon, Z=6, A=12, all cgs ===========
        Z = 6
        A = 12
        Ye = Z / A
        opacity = (56 / (15 * np.sqrt(3))) * (statC_e ** 6 / (c * h * kB ** 2)) * (Z**2 / (mH * A)) / (T ** 2)
        lambda_R_log = np.log((20 * np.sqrt(3) / 14) * ((c * h * sigma * (kB ** 2) * mH / statC_e**6) * (A / (Z ** 2)))) +  5 * np.log(T) - np.log(rho)
        a0 = 5.55e-2 * (rho ** (1/3))
        I1 = 2 * np.pi * np.log(1 + a0 ** 2)
        mu = mp / (mH * Ye)
    
        lambda_T_log = np.log((np.pi/8) * ((h**3 * kB ** 2) / (statC_e ** 4 * me ** 2 * mH)) / (mu * Z) * (T * rho / I1))
        kappa_eff = opacity / (1 + np.exp(lambda_T_log - lambda_R_log))

        return 1 / (kappa_eff * rho) # cm

    def nondeg_mean_free_path(self, T, rho):
        THETA1 = 1.0128
        THETA2 = 1.0823

        Z = 6
        A = 12
        Ye = Z / A

        a0 = 8.45e-7 * T / (rho ** (1/3))
        I1 = 2 * np.pi * np.log(1 + a0 ** 2)
        mu = mp / (mH * Ye)
    
        opacity = (8 * np.pi ** 2 * THETA2 * statC_e ** 6 * h ** 2 * Z ** 2 * rho) / (315 * np.sqrt(3) * THETA1 * c * (2 * np.pi * me) ** (3/2) * mH ** 2 * kB ** (7/2) * A * mu * T ** (7/2))
        lambdaR = (105 * np.sqrt(3) * THETA1 * (c ** 2) * (2 * np.pi * me) ** (3/2) * sigma * (mH ** 2) * (kB ** (7/2)) * A * mu * T ** (13/2)) / (2 * (np.pi ** 2) * THETA2 * (statC_e ** 6) * (h ** 2) * (Z * rho) ** 2)
        lambdaC = (2 ** (13/2) * kB ** (7/2) * T ** (5/2)) / (np.pi ** (1/2) * statC_e ** 4 * me ** (1/2) * Z * I1)

        nondeg_kappa_eff = opacity / (1 + lambdaC / lambdaR)

        return 1 / (nondeg_kappa_eff * rho)

    def mean_free_path(self, T, rho):
        if T / ((rho * 1000) ** (2/3)) > 1241:
            return self.nondeg_mean_free_path(T, rho)
        else:
            return self.deg_mean_free_path(T, rho)

    def dTdr(self, r, T, m, rho):
        # r: radius of white dwarf [cm]
        # T: temperature at current r [K]
        # rho: density [g/cc]
        # m: enclosed mass of white dwarf [g]

        L = self.luminosity(E_gamma=proton_energy, m=m)
        l = self.mean_free_path(T, rho)

        return - 3 * L / (64 * np.pi * (r ** 2) * l * (T ** 3) * sigma)

    def photon_pressure(self, E_gamma, r, m, T, rho):
        """
        r: radius [cm]
        m: mass [g]
        ne: electron number density
        nn: nucleon number density
        E_gamma: gamma ray energy [erg]
        T: temperature [K]
        rho: density [g/cc]

        return dPdr from proton decay photon pressure
        """
        # n = np = ne, but with dimension
        # but Ye = Yp since no of proton = electron in atom
        # return dpdr by species
        
        L = self.luminosity(E_gamma, m) # erg /s
        #l = 1 / nsigma # unit = cm
        l = self.mean_free_path(T, rho)
        #print(f"{l:.3e}, {oldl:.3e}")
        #print(f"l: {l:.3e} cm, r: {r/(1000 * 100):.3e} km")

        dudr = 3 * L / (4 * np.pi * l * c * (r ** 2)) # erg cm^-4 s^-1
        return -dudr/3