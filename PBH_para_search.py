"""
Parallel running parameter sweeping in white dwarf

CLEAN_BAD: option to clean broken WD
PBH_list, rho_list, N_list: list of PBH, core density, and N_PBH to be run
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # one core per process
from pathlib import Path

from WhiteDwarf.wd_setup import WhiteDwarf
import WhiteDwarf.source as source
import numpy as np
import pandas as pd
from WhiteDwarf.constants import *
from multiprocessing import Pool

# clean up bad white dwraf
CLEAN_BAD = True          # option to delete stuck/runaway WD
M_CH, R_MAX = 3, 1e5      # Msun, km  (matching convergence.py)
MIN_STEPS = 100           # fewer radial points than this = broken integration, not a star

# Parameters to be swept
PBH_list = [5e14]
rho_list = np.logspace(-1, 3, 17)   
N_list = np.logspace(5, 15, 21)     

# options
MAX_ITER = 20 # max iteration per WD
EPSILON = 5e-4 # epsilon to stop computation
DEBUG = False # print debug message
DR = 1e-6 # step size

def classify(path):
    try:
        df = pd.read_csv(path)
        last = df.iloc[-1]
        if len(df) <= 1 or not np.isfinite(last[["radius", "mass"]]).all():
            return "stuck"
        if last["mass"] > M_CH or last["radius"] > R_MAX:
            return "runaway"
        return "converged"
    except Exception:                       # empty / corrupted file
        return "stuck"


def clean_results(result_dir):
    result_dir = Path(result_dir)
    if not result_dir.exists():
        return
    files = sorted(result_dir.glob("PBH_M*/N*/rho0*.csv"))
    bad = [(f, s) for f in files if (s := classify(f)) != "converged"]
    print(f"Checked {len(files)} files in {result_dir}: "
          f"{len(files) - len(bad)} converged, {len(bad)} to delete")
    for f, status in bad:
        print(f"  deleting [{status}] {f}")
        f.unlink()


def iterate(rhoc_scaled, source_function, max_iter=MAX_ITER, epsilon=EPSILON, DEBUG=DEBUG, dr=DR, **kwargs):
    wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, source=source_function,Z=6, dr=dr, T0=0, **kwargs)

    wd.hydro_integrate(DEBUG=DEBUG)
    if len(wd.R_profile) < MIN_STEPS:
        raise RuntimeError(f"hydro stopped after {len(wd.R_profile)} steps")
    para0 = wd.thermo_integrate(DEBUG=DEBUG)
    print("Finished initial setup.\n")

    for n in range(max_iter):
        print(f"Round {n}: hydro")
        wd.hydro_integrate(DEBUG=DEBUG)
        if len(wd.R_profile) < MIN_STEPS:
            raise RuntimeError(f"round {n}: hydro stopped after {len(wd.R_profile)} steps")
        print(f"Round {n}: thermal")
        para = wd.thermo_integrate(DEBUG=DEBUG)

        eps = np.nanmax(np.abs(para - para0) / para0)
        if eps <= epsilon:
            print(f"Epsilon: {eps:.3e}, Converges, BREAK")
            return wd
        print(f"Finished iter: {n}, epsilon: {eps:.3e} CONTINUE")
        para0 = para

    print("===== DOESN'T CONVERGE =====")


def has_spectra(PBH_M):
    folder = f"./blackhawk_v2.3/results/Hazma/M{PBH_M:.0e}"
    return all(os.path.exists(f"{folder}/instantaneous_{t}_spectra.txt") for t in ("primary", "secondary"))


def run_one(job):
    # one (PBH mass, N, rho) point; runs in its own process
    PBH_M, N, rho = job
    PBH = source.singlePBH(PBH_M, N=N)
    #PBH = source.No_source()
    new_dir = os.path.join('result', PBH.source_type, f'PBH_M{PBH_M:.1e}/N{N:.1e}/')
    os.makedirs(new_dir, exist_ok=True)
    output_path = os.path.join(new_dir, f'rho0{rho:.3e}.csv')
    if os.path.exists(output_path):
        print(f"Skipping rho={rho:.3e}, file already exists")
        return
    print(f"Working on rho: {rho}, M: {PBH_M}, N: {N:.1e}")
    try:
        wd = iterate(rho, PBH, DEBUG=False,)
    except Exception as e:
        print(f"ERROR at N={N:.1e}, rho={rho:.3e}: {e}")
        return
    if wd is not None:
        R_km, M_sun = wd.rbar2r(wd.R_profile[-1]), wd.mbar2m(wd.M_profile[-1])
        print(f"White dwarf with PBH: {R_km:.3e} km, {M_sun:.3e} Msolar")
        wd.write_csv(output_path)


if __name__ == "__main__":

    # delete stuck/runaway results for smaller timestep
    if CLEAN_BAD:
        source_type = source.singlePBH(PBH_list[0], N=N_list[0]).source_type
        clean_results(os.path.join("result", source_type))

    missing = [M for M in PBH_list if not has_spectra(M)]
    if missing:
        print("No BlackHawk spectra for these masses, skipping them:", ", ".join(f"{M:.0e}" for M in missing))
    PBH_list = [M for M in PBH_list if M not in missing]

    jobs = [(M, N, rho) for M in PBH_list for N in N_list for rho in rho_list]
    n_proc = max(os.cpu_count() - 2, 1)
    print(f"Running {len(jobs)} jobs on {n_proc} processes")

    with Pool(n_proc) as pool:
        pool.map(run_one, jobs, chunksize=1)