import numpy as np
from WhiteDwarf.constants import *
from WhiteDwarf.rk4 import rk4
import matplotlib.pyplot as plt
import helmeos
from scipy.interpolate import interp1d
import csv
from scipy.optimize import brentq

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, source,  Z=6, T0=0, dr=1e-5, r0=1e-3, A=12, **kwargs):
        self.Ye = Ye
        if source is None:
            print("Please input energy source!")
            return
        else:
            self.source = source
            print(f"Using source: {self.source.source_type}")

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
        string = ""
        if hasattr(self,'M_profile') and hasattr(self, 'R_profile'):
            string += f"Mass: {self.mbar2m(self.mbar2m(self.M_profile[-1])):.3f} Msolar, Radius: {self.rbar2r(self.R_profile[-1]):.3f} km "
        if hasattr(self, "T_profile"):
            string += f"Surface Temperature: {self.T_profile[-1]:.3e}, Core Temperature: {self.T_profile[0]:.3e}"

        return string
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
        
        L = self.source.luminosity(r=r, m=m, rho=rho, T=T, E_fermi=self.fermi_energy(rho), E_el_int=self.get_elec_int(T=T, rho=rho)) # erg /s
        l = self.mean_free_path(T, rho) # cm

        dudr = 3 * L / (4 * np.pi * l * c * (r ** 2)) # erg cm^-4 s^-1
        return -dudr/3

    # =================== Electron ===================
    
    def elec_int(self, T):
        # electronic internal energy for finite temperature
        # compute x under finite T
        import numpy as np


    def get_x(self, n, T=0.0):
        a = 8 * np.pi * me**3 * c**3 / (3 * h**3)
        x0 = (n / a) ** (1.0 / 3.0)

        if T == 0:
            return x0

        theta = kB * T / (me * c**2)
        eps = np.pi**2 * theta**2 * (2 * x0**2 + 1) / (2 * x0**4)

        return x0 * (1 - eps / 3.0)
    
    def get_elec_int(self, rho, T):
        n = rho / (mp / self.Ye)
        if n <= 0 or T <= 0:
            return 0.0

        a = 8 * np.pi * me**3 * c**3 / (3 * h**3)
        x0 = (n / a) ** (1.0 / 3.0)
        theta = kB * T / (me * c**2)

        if theta > x0:
            E_density = 1.5 * n * kB * T       # erg/cm^3
        else:
            x = self.get_x(n, T)
            f = lambda x: x * (2*x**2 - 3) * np.sqrt(x**2+1) + 3*np.arcsinh(x)
            g = lambda x: 8*x**3 * (np.sqrt(x**2+1) - 1) - f(x)
            A = np.pi * me**4 * c**5 / (3*h**3)
            E_density = A * g(x) * (
                1 + 4*np.pi**2 * theta**2 *
                ((3*x**2+1)*np.sqrt(x**2+1) - (2*x**2+1)) / (x*g(x))
            )                                    # erg/cm^3

        E_per_particle_erg = E_density / n       # erg
        return E_per_particle_erg / GEV_TO_ERG     

    def fermi_energy(self, rho):
        """
        input: rho in g/cm3
        output: fermi energy in GeV
        """
        n_e = rho * self.Ye / mp
        p_F = hbar * (3. * np.pi**2 * n_e) ** (1/3)
        x = p_F / (me * c)
        E_F = me * c**2 * (np.sqrt(1. + x**2) - 1.)
        return E_F / GEV_TO_ERG # return GeV
    
    def get_proton_pressure(self, rb, m, rho, T):
        # Input are all dimensionless parameters
        proton_pressure = self.photon_pressure(r=rb*self.R0, rho=rho*self.rho0, m=m*self.M0, T=T) * self.R0 / (self.rho0 * c ** 2)
        return proton_pressure
    
    # =================== TOV ===================
    
    def TOV(self, r, rho, P, m, T):
        # note that all things are in cgs here
        if r <= (1e-6) * self.R0:
            m = (4 * np.pi / 3) * rho * (r**3) + self.source.get_mass(rho=rho, T=T, r=r)
        t1 = 1 + P / (rho * c ** 2)
        t2 = 1 + 4 * np.pi * (r ** 3) * P / (m * c ** 2)
        t3 = 1 / (1 - 2 * G * m / (r * c ** 2))

        return t1 * t2 * t3

    # =================== Get Derivative ===================

    def get_hydro_deri(self, state, rb): 
        # state = [rho, m, P_photon]
        rho, m, P_photon = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps), max(state[2], 0)
        T = (P_photon * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25 # unit: Kelvin 
        # helmeos
        out = self.table.eos_DT(rho * self.rho0, T, self.A, self.Z)

        # variables calculated in CGS
        Ptot = out['ptot'][0]
        TOV_term = self.TOV(r=rb * self.R0, rho=rho * self.rho0, P=Ptot, m=m * self.M0, T=T)

        if hasattr(self, 'dPdr_photon_interp'):
            try:
                dPdr_photon = self.dPdr_photon_interp(rb)
            except:
                dPdr_photon = 0
        else: # only trigger in first iteration
            if self.P0 != 0:
                dPdr_photon  = self.get_proton_pressure(rb, m, rho, T) # proton_pressure < 0
            else:
                dPdr_photon = 0

        m_src = self.source.get_mass(r=rb * self.R0)      # g; adapt for singlePBH's signature
        m_tot = m * self.M0 + m_src

        TOV_term = self.TOV(r=rb*self.R0, rho=rho*self.rho0, P=Ptot, m=m_tot, T=T)
        dPdr_G = -(G * m_tot * (rho * self.rho0) / (rb * self.R0)**2) * TOV_term
        dPdr_G = - (G * (m * self.M0) * (rho * self.rho0) / (rb * self.R0)**2) * TOV_term

        if T == 0:
            dPdr = dPdr_G
            dPdr_T = 0
        else:
            dPdr_T = (dPdr_photon * self.rho0 * c ** 2 / self.R0) / a / (T ** 3) * out['dpt'][0]
            dPdr = dPdr_G - dPdr_T

        drhodr = dPdr / (out['dpd'][0])

        drhodr_bar = drhodr * self.R0 / self.rho0
        dmdr = rb**2 * max(rho, 0) 
        # return derivative
        return np.array([drhodr_bar, dmdr, dPdr_photon]), np.array([])
    
    def get_thermo_deri(self, state, rb):
        rho, m, P_photon = max(state[0], np.finfo(np.float32).eps), max(state[1], np.finfo(np.float32).eps), max(state[2], 0)
        T = (P_photon * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
        dPdr_T = self.get_proton_pressure(rb, m, rho, T) # proton_pressure < 0
        return np.array([0, 0, dPdr_T]), np.array([T, dPdr_T])
    
    # =================== Integration Loop ===================
    def hydro_integrate(self, DEBUG=False):
        # Initial conditions
        r = self.r0
        if hasattr(self, 'P_photon_profile'):
            state = np.array([self.rhoc_scaled, self.m0, self.P_photon_profile[0]]) # [density, mass, temperature, photon pressure]
        else:
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
                state = rk4(self.get_hydro_deri, dr=dr, rb=r, state=state)
                if state[0] < 0:
                    print("WARNING: negative density")
                    break


                if DEBUG:
                    T = (state[2] * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
                    print(f"Radius (km) {self.rbar2r(r):.3e} | Density (g/cc): {self.rhobar2rho(state[0]):.3e} | Mass (☉): {self.mbar2m(state[1]):.3e} | Photon Pressure: {state[2]:.3e} | T (K): {T:.3e}")
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

        # Boundary condition
        L = self.source.luminosity(r=self.R_profile[-1] * self.R0, 
                                   m=self.M_profile[-1] * self.M0, 
                                   rho=self.rho_profile[0]*self.rho0, 
                                   E_fermi=self.fermi_energy(self.rho_profile[-1]*self.rho0), 
                                   E_el_int=self.get_elec_int(T=self.T_profile[-1] if hasattr(self, "T_profile") else 0, rho=self.rho_profile[0]*self.rho0))
        T = (L / (4 * np.pi * sigma * (self.R_profile[-1] * self.R0) ** 2)) ** 0.25

        rb = self.R_profile[-1]
        print(f"Backward Integration: Surface T: {T:.3e} K")
        
        state = np.array([self.rho_profile[-1], self.M_profile[-1], a * (T**4) / (self.rho0 * c ** 2)]) # [density, mass, T, photon pressure]
        P_photon_history = []
        dPdr_photon_history = []

        while np.round(rb, decimals=7) >= self.r0: # Prevent round of error
            P_photon_history.append(state[-1])
            hydro_deri, _ = self.get_hydro_deri(rb=rb, state=state)
            deri, debug = self.get_thermo_deri(rb=rb, state=state)

            # also evaluate hydro when backward integration for easier convergence
            deri[0] = -hydro_deri[0]
            deri[1] = -hydro_deri[1]
            dPdr_photon_history.append(deri[2])
            state =  rk4(self.get_thermo_deri, dr=-self.dr, rb=rb, state=state)  
            state[0] = self.rho_interp(rb)
            state[1] = self.M_interp(rb)
            if DEBUG:
                T = (state[2] * self.rho0 * c ** 2 / (4 * sigma / (3 * c))) ** 0.25
                print(f"Radius (km) {self.rbar2r(rb):.3e} | Density (g/cc): {self.rhobar2rho(state[0]):.3e} | Mass (☉): {self.mbar2m(state[1]):.3e} | Photon Pressure: {state[2]:.3e} | T (K): {T:.3e} | T/rho^2/3: {T/((self.rho0 * state[0] / 1000)**2/3):.3e}")

            rb += -self.dr

        if self.source.source_type == 'GaussianPBH':
            total_PBH = self.source._N_enclosed(self.R0 * self.R_profile[-1])
            print(f"The total number of enclosed PBH is {total_PBH:.3e}")

        self.dPdr_photon_profile = np.array(dPdr_photon_history[::-1])
        self.P_photon_profile = np.array(P_photon_history[::-1])
        self.T_profile = (self.P_photon_profile * (self.rho0 * c ** 2) / a) ** 0.25
        self.P0 = self.P_photon_profile[-1]

        r = self.R_profile[-1]

        self.dPdr_photon_interp = interp1d(self.R_profile, self.dPdr_photon_profile, bounds_error=False, fill_value="extrapolate")
        self.P_photon_interp = interp1d(self.R_profile, self.P_photon_profile, bounds_error=False, fill_value="extrapolate")
        return np.array([r, self.rho_profile[0], self.M_profile[0], self.T_profile[0]])
    
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
    def write_csv(self, filename="profiles.csv"):
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
