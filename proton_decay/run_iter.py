from wd_setup import WhiteDwarf
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from constants import *
def iterate(rhoc_scaled, max_iter=50, epsilon=1e-4):
    wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, Z=6, k=2e-26, P0=0, T0=0)
    wd.hydro_integrate(DEBUG=False)
    para0 = wd.thermo_integrate(DEBUG=False)
    for n in range(max_iter):
        wd.hydro_integrate(DEBUG=False)
        para = wd.thermo_integrate(DEBUG=False)
        print(f"r (km) {wd.rbar2r(para[0]):.3e}, rho (g/cc): {wd.rhobar2rho(para[1]):.3e}, M (Msolar): {wd.mbar2m(para[2]):.3e}, T (K): {(para[2] * wd.rho0/ a) ** 0.25:.3e}, Luminosity: {wd.p_decay.luminosity(proton_energy, wd.M_profile[-1] * wd.M0):.3e}")
        
        eps = np.sum((para-para0) ** 2) / len(para)
        if (eps <= epsilon).all():
            print(f"Epsilon: {np.mean(eps):.3e}, Converges, BREAK")
            return wd
        else:
            print(f"Finished iter: {n}, epsilon: {np.mean(eps):.3e} CONTINUE")
            para0 = para

    print("===== DOESN'T CONVERGE =====")
    return wd

wd = iterate(1e-1)