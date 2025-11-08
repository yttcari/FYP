import numpy as np
from constants import *
from decay import decay
from rk4 import rk4
import matplotlib.pyplot as plt

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, Z, k):
        self.Ye = Ye
        self.rhoc_scaled = rhoc_scaled
        self.p_decay = decay(k=k, Z=Z, Ye=Ye)
        self.k = k

        # scale
        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.n0 = 5.89e29

    def __repr__(self):
        if hasattr(self,'mass') and hasattr(self, 'radius'):
            return f"Ye: {self.Ye}, k: {self.k} s-1, Core density: {self.rhobar2rho(self.rhoc_scaled):.3e} g/cc, Mass: {self.mbar2m(self.mass):.3f} Msolar, Radius: {self.rbar2r(self.radius):.3f} km"
        else:
            return rf"Ye: {self.Ye}, k: {self.k} s-1, Core density: {self.rhobar2rho(self.rhoc_scaled):.3e} g/cc"
    
    def _gamma(self, x):
        return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    
    def _x(self, rhob):
        return (rhob) ** (1/3)
    
    def get_pressure(self, x):
        P = self.Ye * me * (x * np.sqrt(x ** 2 + 1) * (2 * x ** 2 -3) + 3 * np.arctanh(x / np.sqrt(x ** 2 + 1))) / (8 * mp)
        return P
    
    def get_proton_pressure(self, rb, m, rho):
        # all dimensionless
        nn = rho * self.rho0 / mp # nucleon density, cm^-3
        ne = self.Ye * nn # electron/proton density, unit = cm-^3, Yp=Ye
        proton_pressure = self.p_decay.photon_pressure(E_gamma=proton_energy, r=rb*self.R0, m=m*self.M0, ne=ne, nn=nn) * self.R0 / self.rho0 
        return proton_pressure
    
    def TOV(self, rb, rho, P, m):
        if rb <= 1e-6:
            m = (1/3) * rho * (rb**3)
        
        t0 = (m * rho / rb**2)
        t1 = (1 + P/rho)
        t2 = (1 + (rb**3 * P)/m)
        t3 = 1 / (1 - (2 * m * self.Ye * me / mp)/rb)

        return -t0 * t1 * t2 * t3

    def get_derivative(self, state, rb): 
        # state = [rho, m]
        rho, m = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps)
        
        x = self._x(rho)
        gamma = self._gamma(x)
        P = max(self.get_pressure(x=x), 0)

        dmdr = rb**2 * max(rho, 0) 

        TOV_term = self.TOV(rb=rb, rho=rho, P=P, m=m)
        proton_pressure = self.get_proton_pressure(rb, m, rho)
        dPdr = TOV_term - proton_pressure 

        drhodr = dPdr / max(gamma, np.finfo(np.float32).eps) 

        #return np.array([drhodr, dmdr]), [t1, t2, t3, P, pion_photon_pressure * 2 + positron_photon_pressure * 2]
        return np.array([drhodr, dmdr]), [TOV_term, proton_pressure]
    
    def integrate(self, DEBUG=False):
        # Initial conditions
        r = 1e-3
        state = np.array([self.rhoc_scaled, (1/3) * self.rhoc_scaled * (r ** 3)]) # [density, mass]

        R_history = []
        M_history = []
        rho_history = []

        while state[0] > 0:
            dr = 5e-4 * r

            R_history.append(r)
            rho_history.append(state[0])
            M_history.append(state[1])
            
            deri, debug = self.get_derivative(rb=r, state=state)

            state = rk4(self.get_derivative, dr=dr, rb=r, state=state)

            if DEBUG:
                print(f"dr (km): {self.rbar2r(dr):.3e} | Radius (km) {self.rbar2r(r):.3e} | Density (g/cc): {self.rhobar2rho(state[0]):.3e} | Mass (☉): {self.mbar2m(state[1]):.3e} | TOV: {debug[0]:.3e} | Proton Pressure: {debug[1]:.3e} | drhodr: {deri[0]:.3e}")
            r += dr

        self.R_profile = np.array(R_history)
        self.M_profile = np.array(M_history)
        self.rho_profile = np.array(rho_history)

        self.mass = M_history[-1]
        self.radius = R_history[-1]

    # Plotting
    def plot_profile(self, type, ax=None, xscale=None, yscale=None, title=None, label=''):
        if ax is None:
            fig, ax = plt.subplots()

        if type == 'rho':
            ax.plot(self.rbar2r(self.R_profile), self.rhobar2rho(self.rho_profile), label=label)
            ax.set_xlabel('Radius (km)')
            ax.set_ylabel(r'Density (g/cm$^3$)')
        elif type == 'M':
            ax.plot(self.rbar2r(self.R_profile), self.mbar2m(self.M_profile), label=label)
            ax.set_xlabel('Radius (km)')
            ax.set_ylabel(r'Mass ($M_\odot$)')
            ax.set_title(f"Total Mass: {self.mbar2m(self.M_profile[-1]):.3f}"+r'$M_{\odot}$')
        else:
            raise ValueError("Variables does not exist.")

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if title is not None:
            ax.set_title(title)

        return ax

    def plot(self, **kwargs):
        fig, ax = plt.subplots(1, 2, **kwargs)

        self.plot_profile('rho', ax=ax[0])
        self.plot_profile('M', ax=ax[1])
        plt.tight_layout()

    # Unit conversion
    def rbar2r(self, rbar):
        return rbar * self.R0 / (100 * 1000)

    def rhobar2rho(self, rhobar):
        return rhobar * self.rho0
    
    def mbar2m(self, mbar):
        return mbar * self.M0 / M_SOLAR