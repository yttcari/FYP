import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator, griddata

root = "../result/SinglePBH"
pattern = os.path.join(root, "PBH_M*", "N*", "rho0*.csv")
paths = sorted(glob.glob(pattern))
m_re = re.compile(r"^PBH_M([\d.eE+-]+)$")
n_re = re.compile(r"^N([\d.eE+-]+)$")
file_re = re.compile(r"^rho0([\d.eE+-]+)\.csv$")

records = []  # list of (M, N, rho, surface_temperature, core_temperature)

m_re = re.compile(r"^PBH_M([\d.eE+-]+)$")
n_re = re.compile(r"^N([\d.eE+-]+)$")
file_re = re.compile(r"^rho0([\d.eE+-]+)\.csv$")

for path in paths:
    filename = os.path.basename(path)
    n_dir = os.path.basename(os.path.dirname(path))
    m_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))

    m_match = m_re.match(m_dir)
    n_match = n_re.match(n_dir)
    file_match = file_re.match(filename)

    if not m_match or not n_match or not file_match:
        print(f"WARNING: could not parse M/N/rho from '{path}', skipping")
        continue

    M = float(m_match.group(1))
    N = float(n_match.group(1))
    rho = float(file_match.group(1))

    df = pd.read_csv(path)

    if df.empty:
        print(f"WARNING: '{path}' is empty, skipping")
        continue

    surface_row = df.loc[df["radius"].idxmax()]
    T_surface = surface_row["temperature"]

    core_row = df.loc[df["radius"].idxmin()]
    T_core = core_row["temperature"]
    rho_core = core_row["density"]

    records.append((M, N, rho_core, T_surface, T_core))

df_all = pd.DataFrame(records,columns=["M", "N", "rho", "T_surface", "T_core"])

def plot_heatmap(fig, ax, rho_values, N_values, temp, sub,
                 title,
                 xlabel="Number of PBH ($N$)",
                 ylabel=r"Center Density ($\rho$) [g/cc]",
                 xlimit=None,
                 scatter=False,
                 M=None):

    # build grid
    T_grid = np.full((len(rho_values), len(N_values)), np.nan)

    for _, row in sub.iterrows():
        i = np.searchsorted(rho_values, row["rho"])
        j = np.searchsorted(N_values, row["N"])
        T_grid[i, j] = row[temp]

    # missing points are left as NaN (not filled)
    nan_mask = np.isnan(T_grid)

    if nan_mask.any():
        m_str = f"M={M:.1e} " if M is not None else ""
        print(f"WARNING: {m_str}has {nan_mask.sum()}/{T_grid.size} missing grid points, leaving them blank")

    # interpolation
    n_fine = 200

    log_rho = np.log10(rho_values)
    log_N = np.log10(N_values)
    log_T = np.log10(T_grid)

    interp = RegularGridInterpolator((log_rho, log_N), log_T, method="linear", bounds_error=False, fill_value=None)

    # restrict the N axis to the visible window
    lo, hi = log_N.min(), log_N.max()
    if xlimit is not None:
        lo_win = max(lo, np.log10(xlimit[0]))
        hi_win = min(hi, np.log10(xlimit[1]))
        if lo_win < hi_win:          # ignore windows outside the data
            lo, hi = lo_win, hi_win

    N_fine = np.logspace(lo, hi, n_fine)
    rho_fine = np.logspace(log_rho.min(), log_rho.max(), n_fine)

    rho_fine_grid, N_fine_grid = np.meshgrid(rho_fine, N_fine, indexing="ij")

    query_points = np.column_stack([np.log10(rho_fine_grid).ravel(), np.log10(N_fine_grid).ravel()])

    T_fine = 10.0 ** interp(query_points).reshape(rho_fine_grid.shape)

    # color scale
    visible = T_fine[np.isfinite(T_fine)]

    norm = LogNorm(vmin=visible.min(), vmax=visible.max()) if visible.size and visible.min() > 0 else None

    # plot
    mesh = ax.pcolormesh(N_fine, rho_fine, T_fine, shading="auto", norm=norm, rasterized=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xlimit is not None:
        ax.set_xlim(*xlimit)

    if scatter:
        N_grid_pts, rho_grid_pts = np.meshgrid(N_values, rho_values)
        ax.scatter(N_grid_pts, rho_grid_pts, c="white", s=10, edgecolors="k", linewidths=0.5, zorder=3)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.solids.set_rasterized(True)   # avoids white seams in PGF/PDF

    return ax

M = 5e14

if True:
    sub = df_all[df_all["M"] == M]

    N_values = np.sort(sub["N"].unique())
    rho_values = np.sort(sub["rho"].unique())

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    plot_heatmap(fig, ax[0], rho_values, N_values, "T_core", sub, title="Core Temperature [K]", ylabel=r"Center Density ($\rho$) [g/cc]", xlimit=(1e5, 1e14))

    plot_heatmap(fig, ax[1], rho_values, N_values, "T_surface", sub, title="Surface Temperature [K]", ylabel="", xlimit=(1e5, 1e14))

    plt.tight_layout()
    plt.show()
    fig.savefig("./img/M5e14_heatmap.png", dpi=300)