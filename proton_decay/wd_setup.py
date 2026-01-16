import numpy as np
from constants import *
from decay import decay
from rk4 import rk4
import matplotlib.pyplot as plt
import helmeos
from scipy.interpolate import interp1d

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, Z, k, T0=0, dr=1e-3, r0=1e-3, A=12):
        self.Ye = Ye
        self.p_decay = decay(k=k, Z=Z, Ye=Ye)
        self.k = k
        self.Z = Z # charge of nucleus
        self.A = A # atomic weight in u
        
        self.dr = dr # integration stepsize in dimensionless form
        self.r0 = r0 # initial step size
        
        self.rhoc_scaled = rhoc_scaled        
        self.m0 = (1/3) * self.rhoc_scaled * (self.r0 ** 3)
 
        # scale
        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.n0 = 5.89e29

        self.P0 = T0 ** 4 * (4 * sigma / (3 * c)) / self.rho0 / c ** 2

        print("Finish setting up white dwarf parameters \n")

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
        P = self.Ye * me * (x * np.sqrt(x ** 2 + 1) * (2 * x ** 2 -3) + 3 * np.sinh(x)) / (8 * mp)
        return P
    
    def get_thermal_pressure(self, x, T):
        f = (x * np.sqrt(x ** 2 + 1) * (2 * x ** 2 -3) + 3 * np.sinh(x)) 

        P = self.get_pressure(x)

        return P * (4 * np.pi ** 2 * (kB * T / (me * c ** 2)) ** 2 * (x * np.sqrt(x**2 + 1)) / (f))
    
    def get_proton_pressure(self, rb, m, rho, T):
        # all dimensionless
        nn = rho * self.rho0 / mp # nucleon density, cm^-3
        ne = self.Ye * nn # electron/proton density, unit = cm-^3, Yp=Ye
        proton_pressure = self.p_decay.photon_pressure(E_gamma=proton_energy, r=rb*self.R0, m=m*self.M0, rho=rho*self.rho0, T=T) * self.R0 / (self.rho0 * c ** 2)
        return proton_pressure
    
    def TOV_bar(self, rb, rho, P, m):
        # for dimensionless calculation
        if rb <= 1e-6:
            m = (1/3) * rho * (rb**3)
        
        #t0 = (m * rho / rb**2)
        t1 = (1 + P/rho)
        t2 = (1 + (rb**3 * P)/m)
        t3 = 1 / (1 - (2 * m * self.Ye * me / mp)/rb)

        return t1 * t2 * t3
    
    def TOV(self, r, rho, P, m):
        # note that all things are in cgs here
        if r <= (1e-6) * self.R0:
            m = (4 * np.pi / 3) * rho * (r**3)
        t1 = 1 + P / (rho * c ** 2)
        t2 = 1 + 4 * np.pi * (r ** 3) * P / (m * c ** 2)
        t3 = 1 / (1 - 2 * G * m / (r * c ** 2))

        return t1 * t2 * t3


    def get_hydro_deri(self, state, rb): 
        # state = [rho, m, P_photon]
        rho, m, P_photon = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps), (state[2])
        T = (P_photon * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25 # unit: Kelvin 
        # helmeos
        out = helmeos.eos_DT(rho * self.rho0, T, self.A, self.Z)

        # variables calculated in CGS
        Ptot = out['ptot'][0]
        TOV_term = self.TOV(r=rb * self.R0, rho=rho * self.rho0, P=Ptot, m=m * self.M0)

        if hasattr(self, 'dPdr_photon_interp'):
            try:
                dPdr_photon = self.dPdr_photon_interp(rb)
            except:
                dPdr_photon = 0
        else:
            if self.P0 != 0:
                dPdr_photon  = self.get_proton_pressure(rb, m, rho, T) # proton_pressure < 0
            else:
                dPdr_photon = 0

        dPdr_G = - (G * (m * self.M0) * (rho * self.rho0) / (rb * self.R0)**2) * TOV_term

        if T == 0:
            dPdr = dPdr_G
        else:
            print(dPdr_photon)
            dPdr_T = (dPdr_photon * self.rho0 * c ** 2 / self.R0) / a / (T ** 3) * out['dpt'][0]
            dPdr = dPdr_G - dPdr_T

        drhodr = dPdr / (out['dpd'][0])

        drhodr_bar = drhodr * self.R0 / self.rho0
        dmdr = rb**2 * max(rho, 0) 
        # return derivative
        return np.array([drhodr_bar, dmdr, 0]), np.array([dPdr_G, dPdr_photon ,dPdr, (dPdr_photon * self.rho0 * c ** 2 / self.R0) / a / (T ** 3) * out['dpt'][0], out['dpt'][0], drhodr])
    
    def get_thermo_deri(self, state, rb):
        rho, m, P_photon = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps), max(state[2], np.finfo(np.float32).eps)
        T = (P_photon * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
        dPdr_T = self.get_proton_pressure(rb, m, rho, T) # proton_pressure < 0
        return np.array([0, 0, dPdr_T]), np.array([T, dPdr_T])
    
    def hydro_integrate(self, DEBUG=False):
        # Initial conditions
        r = self.r0
        state = np.array([self.rhoc_scaled, self.m0, self.P0]) # [density, mass, temperature, photon pressure]
        dr = self.dr

        R_history = []
        M_history = []
        rho_history = []
        debug_history = []

        while state[0] > 1e-5:
            try:

                R_history.append(r)
                rho_history.append(state[0])
                M_history.append(state[1])       
                
                deri, debug = self.get_hydro_deri(rb=r, state=state)
                debug_history.append(debug)
                if hasattr(self, 'P_photon_interp'):
                    try:
                        if r < self.R_profile[-1]:
                            # Prevent interpolation error when new hydro cycle integrate out of the old radius
                            state[2] = self.P_photon_interp(r)
                        else:
                            state[2] = self.P_photon_profile[-1]
                    except:
                        state[2] = self.P0
                else:
                    state[2] = self.P0

                state = rk4(self.get_hydro_deri, dr=dr, rb=r, state=state)
                if state[0] < 0:
                    print("WARNING: negative density")
                    break
                if state[1] < 0:
                    print("WARNING: negative pressure")
                    state[1] = 0

                if DEBUG:
                    T = (state[2] * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
                    print(f"Radius (km) {self.rbar2r(r):.3e} | Density (g/cc): {self.rhobar2rho(state[0]):.3e} | Mass (☉): {self.mbar2m(state[1]):.3e} | Photon Pressure: {state[2]:.3e} | T (K): {T:.3e} | T/rho^2/3: {T/((self.rho0 * state[0] / 1000)**2/3):.3e} | dPdr_G: {debug[0]:.3e} | dPdr: {debug[2]:.3e} | dPdr_T: {debug[3]:.3e}")
                r += dr

            except StopIteration:
                break

        self.R_profile = np.array(R_history)
        self.M_profile = np.array(M_history)
        self.rho_profile = np.array(rho_history)
        self.debug_profile = np.array(debug_history)

        self.rho_interp = interp1d(self.R_profile, self.rho_profile, bounds_error=False, fill_value="extrapolate")
        self.M_interp = interp1d(self.R_profile, self.M_profile, bounds_error=False, fill_value="extrapolate")

    def thermo_integrate(self, DEBUG=False):
        # do the backward integration
        L = self.p_decay.luminosity(proton_energy, self.M_profile[-1] * self.M0) 
        rb = self.R_profile[-1]
        T_surface = (L / (4 * np.pi * sigma * (self.R_profile[-1] * self.R0) ** 2)) ** 0.25
        print(f"Backward Integration: Surface T: {T_surface:.3e} K")
        
        state = np.array([self.rho_profile[-1], self.M_profile[-1], a * (T_surface**4) / (self.rho0 * c ** 2)]) # [density, mass, T, photon pressure]
        
        P_photon_history = []
        dPdr_photon_history = []

        while np.round(rb, decimals=7) >= self.r0: # Prevent round of error
            P_photon_history.append(state[-1])
            deri, debug = self.get_thermo_deri(rb=rb, state=state)
            dPdr_photon_history.append(deri[2])
            state =  rk4(self.get_thermo_deri, dr=-self.dr, rb=rb, state=state)  
            state[0] = self.rho_interp(rb)
            state[1] = self.M_interp(rb)
            
            if DEBUG:
                T = (state[2] * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
                print(f"Radius (km) {self.rbar2r(rb):.3e} | Density (g/cc): {self.rhobar2rho(state[0]):.3e} | Mass (☉): {self.mbar2m(state[1]):.3e} | Photon Pressure: {state[2]:.3e} | T (K): {T:.3e} | T/rho^2/3: {T/((self.rho0 * state[0] / 1000)**2/3):.3e}")

            rb += -self.dr

        self.dPdr_photon_profile = np.array(dPdr_photon_history)
        self.P_photon_profile = np.array(P_photon_history[::-1])
        self.T_profile = (self.P_photon_profile * (self.rho0 * c ** 2) / a) ** 0.25
        self.P0 = self.P_photon_profile[-1]

        r = self.R_profile[-1]

        self.dPdr_photon_interp = interp1d(self.R_profile, self.dPdr_photon_profile, bounds_error=False, fill_value="extrapolate")
        self.P_photon_interp = interp1d(self.R_profile, self.P_photon_profile, bounds_error=False, fill_value="extrapolate")
        return np.array([r, self.rho_profile[-1], self.M_profile[-1], self.P_photon_profile[-1]])

    # Plotting
    def plot_profile(self, type, ax=None, xscale=None, yscale=None, title=None, label='', ylim=None):
        if ax is None:
            fig, ax = plt.subplots()

        if type == 'rho':
            #ax.plot(self.rbar2r(self.R_profile), self.rho_interp(self.R_profile), label=label)
            ax.plot(self.rbar2r(self.R_profile), self.rhobar2rho(self.rho_profile), label=label)
            ax.set_xlabel('Radius (km)')
            ax.set_ylabel(r'Density (g/cm$^3$)')
        elif type == 'M':
            ax.plot(self.rbar2r(self.R_profile), self.mbar2m(self.M_profile), label=label)
            ax.set_xlabel('Radius (km)')
            ax.set_ylabel(r'Mass ($M_\odot$)')
            ax.set_title(f"Total Mass: {self.mbar2m(np.max(self.M_profile)):.3f}"+r'$M_{\odot}$')
        elif type == 'T':
            ax.plot(self.rbar2r(self.R_profile), self.T_profile, label=label)
            ax.set_xlabel('Radius (km)')
            ax.set_ylabel(r'Temperature (K)')
        else:
            raise ValueError("Variables does not exist.")

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if title is not None:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylimit(ylim)

        return ax

    def plot(self, **kwargs):
        if hasattr(self, 'T_profile'):
            fig, ax = plt.subplots(1, 3, **kwargs)

            self.plot_profile('rho', ax=ax[0], title='Density')
            self.plot_profile('M', ax=ax[1])

            self.plot_profile('T', ax=ax[2], title='Temperature')
        else:
            fig, ax = plt.subplots(1, 2, **kwargs)

            self.plot_profile('rho', ax=ax[0], title='Density')
            self.plot_profile('M', ax=ax[1])
        plt.tight_layout()

    # Unit conversion
    def rbar2r(self, rbar):
        return rbar * self.R0 / (100 * 1000)

    def rhobar2rho(self, rhobar):
        return rhobar * self.rho0
    
    def mbar2m(self, mbar):
        return mbar * self.M0 / M_SOLAR