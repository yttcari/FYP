import numpy as np
from constants import *
from rk4 import rk4
import matplotlib.pyplot as plt
import helmeos
from scipy.interpolate import interp1d
import csv

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, Z, source, T0=0, dr=1e-3, r0=1e-3, A=12):
        self.Ye = Ye
        if source is None:
            print("Please input energy source!")
            return
        else:
            self.source = source
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

        self.table = helmeos.HelmTable()

        print("Finish setting up white dwarf parameters \n")

    def __repr__(self):
        if hasattr(self,'mass') and hasattr(self, 'radius'):
            return f"Ye: {self.Ye}, k: {self.k} s-1, Core density: {self.rhobar2rho(self.rhoc_scaled):.3e} g/cc, Mass: {self.mbar2m(self.mass):.3f} Msolar, Radius: {self.rbar2r(self.radius):.3f} km"
        else:
            return rf"Ye: {self.Ye}, k: {self.k} s-1, Core density: {self.rhobar2rho(self.rhoc_scaled):.3e} g/cc"
    
    # =================== Radiation ===================

    def deg_mean_free_path(self, T, rho):
        # Assumed Carbon, Z=6, A=12, all cgs 
        opacity = (56 / (15 * np.sqrt(3))) * (statC_e ** 6 / (c * h * kB ** 2)) * (self.Z**2 / (mH * self.A)) / (T ** 2)
        lambda_R_log = np.log((20 * np.sqrt(3) / 14) * ((c * h * sigma * (kB ** 2) * mH / statC_e**6) * (self.A / (self.Z ** 2)))) +  5 * np.log(T) - np.log(rho)
        a0 = 5.55e-2 * (rho ** (1/3))
        I1 = 2 * np.pi * np.log(1 + a0 ** 2)
        mu = mp / (mH * self.Ye)
    
        lambda_T_log = np.log((np.pi/8) * ((h**3 * kB ** 2) / (statC_e ** 4 * me ** 2 * mH)) / (mu * self.Z) * (T * rho / I1))
        kappa_eff = opacity / (1 + np.exp(lambda_T_log - lambda_R_log))

        return 1 / (kappa_eff * rho) # cm

    def nondeg_mean_free_path(self, T, rho):
        THETA1 = 1.0128
        THETA2 = 1.0823

        a0 = 8.45e-7 * T / (rho ** (1/3))
        I1 = 2 * np.pi * np.log(1 + a0 ** 2)
        mu = mp / (mH * self.Ye)
    
        opacity = (8 * np.pi ** 2 * THETA2 * statC_e ** 6 * h ** 2 * self.Z ** 2 * rho) / (315 * np.sqrt(3) * THETA1 * c * (2 * np.pi * me) ** (3/2) * mH ** 2 * kB ** (7/2) * self.A * mu * T ** (7/2))
        lambdaR = (105 * np.sqrt(3) * THETA1 * (c ** 2) * (2 * np.pi * me) ** (3/2) * sigma * (mH ** 2) * (kB ** (7/2)) * self.A * mu * T ** (13/2)) / (2 * (np.pi ** 2) * THETA2 * (statC_e ** 6) * (h ** 2) * (self.Z * rho) ** 2)
        lambdaC = (2 ** (13/2) * kB ** (7/2) * T ** (5/2)) / (np.pi ** (1/2) * statC_e ** 4 * me ** (1/2) * self.Z * I1)

        nondeg_kappa_eff = opacity / (1 + lambdaC / lambdaR)

        return 1 / (nondeg_kappa_eff * rho)

    def mean_free_path(self, T, rho):
        if T / ((rho * 1000) ** (2/3)) > 1241:
            return self.nondeg_mean_free_path(T, rho)
        else:
            return self.deg_mean_free_path(T, rho)

    def photon_pressure(self, r, rho, m, T):
        """
        r: radius [cm]
        m: mass [g]
        T: temperature [K]
        rho: density [g/cc]

        return dPdr from proton decay photon pressure
        """
        
        L = self.source.luminosity(r=r, m=m) # erg /s
        l = self.mean_free_path(T, rho) # cm

        dudr = 3 * L / (4 * np.pi * l * c * (r ** 2)) # erg cm^-4 s^-1
        return -dudr/3
    
    def get_proton_pressure(self, rb, m, rho, T):
        # Input are all dimensionless parameters
        proton_pressure = self.photon_pressure(r=rb*self.R0, rho=rho*self.rho0, m=m*self.M0, T=T) * self.R0 / (self.rho0 * c ** 2)
        return proton_pressure
    
    # =================== TOV ===================
    
    def TOV(self, r, rho, P, m):
        # note that all things are in cgs here
        if r <= (1e-6) * self.R0:
            m = (4 * np.pi / 3) * rho * (r**3) + self.source.get_mass()
        t1 = 1 + P / (rho * c ** 2)
        t2 = 1 + 4 * np.pi * (r ** 3) * P / (m * c ** 2)
        t3 = 1 / (1 - 2 * G * m / (r * c ** 2))

        return t1 * t2 * t3

    # =================== Get Derivative ===================

    def get_hydro_deri(self, state, rb): 
        # state = [rho, m, P_photon]
        rho, m, P_photon = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps), (state[2])
        T = (P_photon * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25 # unit: Kelvin 
        # helmeos
        out = self.table.eos_DT(rho * self.rho0, T, self.A, self.Z)

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
    
    # =================== Integration Loop ===================
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
                            state[2] = self.P_photon_profile[0]
                    except:
                        state[2] = self.P0
                else:
                    state[2] = self.P0

                state = rk4(self.get_hydro_deri, dr=dr, rb=r, state=state)
                if state[0] < 0:
                    print("WARNING: negative density")
                    break

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
        L = self.source.luminosity(r=self.R_profile[-1] * self.R0, m=self.M_profile[-1] * self.M0) 
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

        self.dPdr_photon_profile = np.array(dPdr_photon_history[::-1])
        self.P_photon_profile = np.array(P_photon_history[::-1])
        self.T_profile = (self.P_photon_profile * (self.rho0 * c ** 2) / a) ** 0.25
        self.P0 = self.P_photon_profile[-1]

        r = self.R_profile[-1]

        self.dPdr_photon_interp = interp1d(self.R_profile, self.dPdr_photon_profile, bounds_error=False, fill_value="extrapolate")
        self.P_photon_interp = interp1d(self.R_profile, self.P_photon_profile, bounds_error=False, fill_value="extrapolate")
        return np.array([r, self.rho_profile[-1], self.M_profile[-1], self.P_photon_profile[-1]])

    # =================== Plotting ===================
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

    # =================== Unit conversion ===================
    def rbar2r(self, rbar):
        return rbar * self.R0 / (100 * 1000)

    def rhobar2rho(self, rhobar):
        return rhobar * self.rho0
    
    def mbar2m(self, mbar):
        return mbar * self.M0 / M_SOLAR
    
    # =================== output ===================
    def export_profiles_to_csv(self, filename="profiles.csv"):
        """
        Export all stored profiles into a single CSV file.
        Each row corresponds to one radial point, with columns for each profile.
        """
        # stack arrays column-wise
        data = np.column_stack([
            self.rbar2r(self.R_profile),
            self.rhobar2rho(self.rho_profile),
            self.mbar2m(self.M_profile),
            self.T_profile,
            self.P_photon_profile,
            self.dPdr_photon_profile
        ])

        # define headers
        headers = ["radius", "density", "mass", "temperature", "Pphoton", 'dPdr']

        # write to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

        print(f"Profiles exported to {filename}")