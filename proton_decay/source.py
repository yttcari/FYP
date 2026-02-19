import numpy as np
from constants import *

hbar = h / (2 * np.pi)

class proton_decay:
    def __init__(self, k, **kwargs):
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

class No_source:
    def __init__(self, **kwargs):
        pass

    def get_mass(self):
        return 0
    
    def luminosity(self, **kwargs):
        return 0

class PBHDistribution:
    def __init__(self, eos, Tmin, Tmax, rho_min, rho_max, atomic_num, mass_num, M_WD=None, alpha=-5/2, f_PBH=0.3, lambda_bondi=1.0, **kwargs):
        """
        Carr PBH distribution inside a WD.

        Parameters
        ----------
        alpha : float
            Power-law index in Carr number profile dn/dM ∝ M^{-alpha}.
        Ntot : float, optional
            Total number of PBHs (number normalization).
        f_PBH : float, optional
            PBH mass fraction relative to WD mass.
        M_WD : float, optional
            Total WD mass (required if f_PBH is used).
        rho_min, rho_max : float
            WD density range from first iteration.
        lambda_bondi : float
            Bondi accretion factor.
        """
        self.alpha = alpha
        self.lambda_bondi = lambda_bondi

        self.eos = eos
        self.mass_num = mass_num
        self.atomic_num = atomic_num

        # Define global M_min, M_max from WD density range
        self.M_min = self.equilibrium_mass(rho_max, c_s=self.eos.eos_DT(rho_min, Tmin, mass_num, atomic_num)['cs'][0])
        self.M_max = self.equilibrium_mass(rho_min,  c_s=self.eos.eos_DT(rho_max, Tmax, mass_num, mass_num)['cs'][0])

        self.Mtot = f_PBH * M_WD
        self.A = self._normalize_mass(self.M_min, self.M_max)
    
    # ---------- Equilibrium mass ----------

    def equilibrium_mass(self, rho, c_s):
        """Equilibrium PBH mass M_eq(rho) (c_s must be assigned externally)."""
        if c_s is None:
            raise ValueError("c_s must be assigned before calling equilibrium_mass.")
        prefactor = hbar * c**4 * c_s**3
        denom = 61440 * np.pi**2 * self.lambda_bondi * G**4 * rho
        return (prefactor / denom)**0.25

    # ---------- Normalization ----------
    def _normalize_mass(self, Mmin, Mmax):
        """Normalize Carr distribution by total mass Mtot."""
        a = self.alpha
        if a == 2:
            return self.Mtot / np.log(Mmax/Mmin)
        return self.Mtot * (2-a) / (Mmax**(2-a) - Mmin**(2-a))

    # ---------- Carr profile ----------

    def dn_dM(self, M):
        return self.A * M**(-self.alpha)

    # ---------- Enclosed mass ----------

    def enclosed_mass(self, rho, T):
        """M_enc = ∫ M dn/dM dM from M_min to M_cut."""
        Mmin = self.M_min
        c_s = self.eos.eos_DT(rho, R, self.mass_num, self.atomic_num)['cs'][0]
        Mcut = np.clip(self.equilibrium_mass(rho, c_s), Mmin, self.M_max)
        a = self.alpha

        if a == 2:
            return self.A * np.log(Mcut/Mmin)
        return self.A * (Mcut**(2-a) - Mmin**(2-a)) / (2-a)

    def get_mass(self, rho, T, **kwargs):
        return self.enclosed_mass(rho, T)

    # ---------- Enclosed luminosity ----------

    def enclosed_luminosity(self, rho, T, **kwargs):
        """L_enc = ∫ (L_acc + L_hawk) dn/dM dM."""
 
        Mmin = self.M_min
        c_s = self.eos.eos_DT(rho, T, self.mass_num, self.atomic_num)['cs'][0]
        Mcut = np.clip(self.equilibrium_mass(rho, c_s), Mmin, self.M_max)
        a = self.alpha
        A = self.A

        K_acc  = 4*np.pi*self.lambda_bondi*G**2*rho*c**2 / c_s**3
        K_hawk = hbar*c**6 / (15360*np.pi*G**2)

        # Accretion term:
        if a == 3:
            L_acc = A*K_acc*np.log(Mcut/Mmin)
        else:
            L_acc = A*K_acc*(Mcut**(3-a) - Mmin**(3-a)) / (3-a)

        # Hawking term: 
        if a == -1:
            L_hawk = A*K_hawk*np.log(Mcut/Mmin)
        else:
            L_hawk = A*K_hawk*(Mcut**(-1-a) - Mmin**(-1-a)) / (-1-a)

        return L_acc + L_hawk

    def luminosity(self, rho, **kwargs):
        return self.enclosed_luminosity(rho, **kwargs)