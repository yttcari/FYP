import numpy as np
from constants import *
from rk4 import rk4

a_rad = 7.5657e-15

class temperature:
    def __init__(self, wd):
        self.wd = wd
        self.rho_profile = wd.rho_profile * wd.rho0
        self.k = wd.k #1/s
        self.R_profile = wd.R_profile * wd.R0
        self.rho_of_r = lambda r: np.interp(r, self.R_profile, self.rho_profile)
        
    def derivative(self, rb, state):
        """
        state = [L, T]
        returns (dstate_dr, extra)
        extra can be None since rk4 ignores it,
        but we keep it to match your signature.
        """
        L, T = state
        rho  = self.rho_of_r(rb)
        kappa = 4.34e24 * self.wd.Z * rho * (T ** (-3.5))

        if rb <= 0.0 or T <= 0.0:
            return np.array([0.0, 0.0]), None

        dL_dr = 4.0 * np.pi * rb**2 * rho * self.k * proton_energy / mp
        nn = rho / mp
        dT_dr = self.wd.p_decay.dTdr(E_gamma=proton_energy, r=rb, L=L, nn=nn, ne=self.wd.Ye * nn, T=T)
        return np.array([dL_dr, dT_dr]), None
    
    def integrate(self, Tc):
        N = len(self.R_profile)

        L = np.zeros(N)
        T = np.zeros(N)

        # central BCs
        T[0] = Tc
        L[0] = 0.0

        for i in range(N-1):
            rb = self.R_profile[i]
            dr = self.R_profile[i+1] - self.R_profile[i]
            state = np.array([L[i], T[i]])
            state_next = rk4(derivative=self.derivative, dr=dr, rb=rb, state=state)
            l, t = state_next
            L[i+1] = l#max(l, 0)
            T[i+1] = t#max(t, 0)

        return L, T
    
    def get_Tc(self, T_eff, T_l, T_r, epsilon=1):
        # just a binary search
        dT = np.inf
        n = 0
        T_profile = 0
        while dT > epsilon:
            T_mid = (T_l + T_r) / 2
            _, T_profile = self.integrate(Tc=T_mid)
            T = T_profile[-1]
            dT = T - T_eff
            if dT > 0: 
                T_r = T_mid
            if dT < 0:
                T_l = T_mid

            dT = abs(dT)
            n+=1
            print(f"Iteration: {n}, dT: {dT}, Tc: {T_mid}")
        return T_profile