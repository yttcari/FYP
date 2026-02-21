from wd_setup import WhiteDwarf
import source
import numpy as np
from tqdm import tqdm
import pandas as pd
from constants import *
import os

output_home_dir = 'PBH/'

def iterate(rhoc_scaled, source_function, max_iter=10, epsilon=5e-4, DEBUG=False, **kwargs):
    wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, source=source_function, Z=6, **kwargs)
    wd.hydro_integrate(DEBUG=DEBUG)
    para0 = wd.thermo_integrate(DEBUG=DEBUG)
    print(f"r (km) {wd.rbar2r(para0[0]):.3e}, rho (g/cc): {wd.rhobar2rho(para0[1]):.3e}, M (Msolar): {wd.mbar2m(para0[2]):.3e}, T (K): {(para0[3] * wd.rho0 * c ** 2/ a) ** 0.25:.3e}, Luminosity: {source_function.luminosity(r=wd.R_profile[0] * wd.R0, m=wd.M_profile[-1] * wd.M0):.3e}")

    print("Finished initial setup.\n")
    for n in range(max_iter):
        print(f"Round {n}: hydro")
        wd.hydro_integrate(DEBUG=DEBUG)
        print(f"Round {n}: thermal")
        para = wd.thermo_integrate(DEBUG=DEBUG)
        print(f"r (km) {wd.rbar2r(para[0]):.3e}, rho (g/cc): {wd.rhobar2rho(para[1]):.3e}, M (Msolar): {wd.mbar2m(para[2]):.3e}, T (K): {(para[3] * wd.rho0/ a) ** 0.25:.3e}, Luminosity: {source_function.luminosity(r=wd.R_profile[0] * wd.R0, m=wd.M_profile[-1] * wd.M0):.3e}")
        

        eps = np.nanmax(np.abs(para-para0)/para0)
        if (eps <= epsilon).all():
            print(f"Epsilon: {np.mean(eps):.3e}, Converges, BREAK")
            return wd
            
        else:
            print(f"Finished iter: {n}, epsilon: {np.mean(eps):.3e} CONTINUE")
            para0 = para

    print("===== DOESN'T CONVERGE =====")

# searching for proton decay
import numpy as np
from scipy.integrate import solve_ivp

# Constants (cgs)
G = 6.67430e-8       # cm^3 g^-1 s^-2
alpha = 5.34e25      # g^3/s

def dMdt(t, M, rho=1e6, cs=1e8, lam=1.0):
    """
    PBH mass evolution: Bondi accretion - Hawking evaporation.
    
    Parameters
    ----------
    t : float
        Time [s]
    M : float
        PBH mass [g]
    rho : float
        Ambient density [g/cm^3]
    cs : float
        Sound speed [cm/s]
    lam : float
        Bondi accretion factor (~1)
    """
    bondi = 4 * np.pi * lam * G**2 * M**2 * rho / cs**3
    hawking = hbar * c ** 6 / (15360 * np.pi * G ** 2 * M**2)
    return bondi - hawking

def evolve_pbh(M0=1e17, t_max=1e15, rho=1e6, cs=1e8, lam=1.0):
    """
    Integrate PBH mass evolution.
    """
    sol = solve_ivp(
        fun=lambda t, M: dMdt(t, M, rho=rho, cs=cs, lam=lam),
        t_span=(0, t_max),
        y0=[M0],
        dense_output=True,
        max_step=1e12
    )
    return sol.t, sol.y[0]
