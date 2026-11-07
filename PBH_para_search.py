import os
os.environ["OMP_NUM_THREADS"] = "1"  # one core per process
from WhiteDwarf.wd_setup import WhiteDwarf
import WhiteDwarf.source as source
import numpy as np
import pandas as pd
from WhiteDwarf.constants import *
from multiprocessing import Pool

def iterate(rhoc_scaled, source_function, max_iter=20, epsilon=5e-4, DEBUG=False, dr=1e-5, **kwargs):
    wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, source=source_function, Z=6, dr=dr, **kwargs)

    # ---- no source: cold star, structure only ----

    wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, source=source_function,
                    Z=6, dr=dr, T0=1e4, **kwargs)      # 1e4 K floor, effectively cold
    if source.source_type == "No_source":    
        wd.hydro_integrate(DEBUG=DEBUG)
        n = len(wd.R_profile)
        wd.T_profile = np.full(n, 1e4)
        wd.P_photon_profile = np.full(n, wd.P0)
        wd.dPdr_photon_profile = np.zeros(n)
        print("No source: hydro only, thermal evolution skipped.")
        return wd
    
    # ---- with source: hydro <-> thermo iteration ----
    wd.hydro_integrate(DEBUG=DEBUG)
    para0 = wd.thermo_integrate(DEBUG=DEBUG)
    print("Finished initial setup.\n")

    for n in range(max_iter):
        print(f"Round {n}: hydro")
        wd.hydro_integrate(DEBUG=DEBUG)
        print(f"Round {n}: thermal")
        para = wd.thermo_integrate(DEBUG=DEBUG)

        eps = np.nanmax(np.abs(para - para0) / para0)
        if eps <= epsilon:
            print(f"Epsilon: {eps:.3e}, Converges, BREAK")
            return wd
        print(f"Finished iter: {n}, epsilon: {eps:.3e} CONTINUE")
        para0 = para

    print("===== DOESN'T CONVERGE =====")"""


def has_spectra(PBH_M):
    folder = f"./blackhawk_v2.3/results/Hazma/M{PBH_M:.0e}"
    return all(os.path.exists(f"{folder}/instantaneous_{t}_spectra.txt") for t in ("primary", "secondary"))


def run_one(job):
    # one (PBH mass, N, rho) point; runs in its own process
    PBH_M, N, rho = job
    #PBH = source.singlePBH(PBH_M, N=N)
    PBH = source.No_source()
    new_dir = os.path.join('result', PBH.source_type, f'PBH_M{PBH_M:.1e}/N{N:.1e}/')
    os.makedirs(new_dir, exist_ok=True)
    output_path = os.path.join(new_dir, f'rho0{rho:.3e}.csv')
    if os.path.exists(output_path):
        print(f"Skipping rho={rho:.3e}, file already exists")
        return
    print(f"Working on rho: {rho}, M: {PBH_M}, N: {N:.1e}")
    try:
        wd = iterate(rho, PBH, DEBUG=False, dr=1e-3, epsilon=1e-4, max_iter=20)
    except Exception as e:
        print(f"ERROR at N={N:.1e}, rho={rho:.3e}: {e}")
        return
    if wd is not None:
        print(f"White dwarf with PBH: {wd.rbar2r(wd.R_profile[-1]):.3e} km, {wd.mbar2m(wd.M_profile[-1]):.3e} Msolar")
        wd.write_csv(output_path)


if __name__ == "__main__":
    # PBH masses need BlackHawk spectra in folder M{mass:.0e}, so use ONE significant digit
    PBH_list = [5e14]
    rho_list = np.logspace(-1, 3, 17)   # 4 points per decade (was 5 points total)
    N_list = np.logspace(5, 15, 21)     # 2 points per decade (was 10 points total)

    missing = [M for M in PBH_list if not has_spectra(M)]
    if missing:
        print("No BlackHawk spectra for these masses, skipping them:", ", ".join(f"{M:.0e}" for M in missing))
    PBH_list = [M for M in PBH_list if M not in missing]

    jobs = [(M, N, rho) for M in PBH_list for N in N_list for rho in rho_list]
    n_proc = max(os.cpu_count() - 2, 1)
    print(f"Running {len(jobs)} jobs on {n_proc} processes")

    with Pool(n_proc) as pool:
        pool.map(run_one, jobs, chunksize=1)
