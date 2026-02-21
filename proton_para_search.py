from wd_setup import WhiteDwarf
import source
import numpy as np
from tqdm import tqdm
import pandas as pd
from constants import *
import os

output_home_dir = 'proton_decay/'

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
k_list = [1e-27, 1e-28, 1e-29, 1e-30]
rho_list = np.logspace(-3, 3, 10)

for k in k_list:
    proton_decay = source.proton_decay(k)
    new_dir = os.path.join(output_home_dir, f'wd_k{k:.1e}/')
    os.makedirs(new_dir, exist_ok=True)
    for rho in tqdm(rho_list):
        output_path = os.path.join(new_dir, f'rho0{rho:.3e}.csv')
        wd = iterate(rho, proton_decay, DEBUG=False)
        if wd is not None:
            print(f"White dwarf with proton decay: {wd.rbar2r(wd.R_profile[-1]):.3e} km, {wd.mbar2m(wd.M_profile[-1]):.3e} Msolar")
            wd.write_csv(output_path)