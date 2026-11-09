"""
Plot converging heatmap for the simulations
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path.cwd().parent          # PBH_WD/
FYP = ROOT.parent                 # folder holding "No source" / "No_source"

SOURCES = {
    "SinglePBH": [ROOT / "result" / "SinglePBH" ],
}

M_CH, R_MAX = 3, 1e5   # Msun, km: beyond this the integration never found a WD surface
RHO_UNIT = 9.79e5 * 2     # g/cm^3: converts dimensionless rho0 in file names to physical density
PAT = re.compile(r"PBH_M([0-9.e+-]+)/N([0-9.e+-]+)/rho0([0-9.e+-]+)\.csv$")


def classify(df):
    last = df.iloc[-1]
    if len(df) <= 1 or not np.isfinite(last[["radius", "mass"]]).all():
        return "stuck"
    if last["mass"] > M_CH or last["radius"] > R_MAX:
        return "runaway"
    return "converged"

# read csv
rows = []
for source, dirs in SOURCES.items():
    print(source)
    for d in dirs:
        for f in sorted(d.glob("PBH_M*/N*/rho0*.csv")) if d.exists() else []:
            m = PAT.search(f.as_posix())
            if m is None:
                continue
            try:
                df = pd.read_csv(f)
                status = classify(df)
                last = df.iloc[-1]
                R, M, n = last["radius"], last["mass"], len(df)
            except Exception:              # empty / corrupted file
                status, R, M, n = "stuck", np.nan, np.nan, 0
            rows.append(dict(source=source, M_PBH=float(m[1]), N=float(m[2]), rho0=float(m[3]),
                             status=status, n_steps=n, R=R, M=M, path=str(f)))

if not rows:
    raise FileNotFoundError(f"No result files found. Check ROOT = {ROOT}")

runs = pd.DataFrame(rows).drop_duplicates(["source", "M_PBH", "N", "rho0"])

# build parameter grid
grid_M, grid_N, grid_rho = (np.sort(runs[c].unique()) for c in ["M_PBH", "N", "rho0"])
sources = list(SOURCES)
full = pd.MultiIndex.from_product([sources, grid_M, grid_N, grid_rho], names=["source", "M_PBH", "N", "rho0"]).to_frame(index=False)
full = full.merge(runs, how="left", on=["source", "M_PBH", "N", "rho0"])
full["status"] = full.status.fillna("missing")
full["converged"] = (full.status == "converged").astype(float)
full["rho0_phys"] = full.rho0 * RHO_UNIT           # physical density, for reference

print(pd.crosstab(full.source, full.status, margins=True))

def to_grid(d, value):
    """rows = rho0 (y), columns = N (x)"""
    return (d.pivot_table(index="rho0", columns="N", values=value, aggfunc="first")
              .reindex(index=grid_rho, columns=grid_N))


# smooth heat map
src = "SinglePBH"
KEEP = [1e14, 5e14]

d_src = full[full.source == src]
masses = [m for m in grid_M if np.any(np.isclose(m, KEEP, rtol=1e-6))]
if not masses:
    raise ValueError(f"None of {KEEP} found. Available M_PBH: {grid_M}")

logN = np.log10(grid_N)
rho_phys = grid_rho * RHO_UNIT                     # y-axis in g/cm^3
X, Y = np.meshgrid(logN, rho_phys)
log_rho_axis = rho_phys.max() / rho_phys.min() > 100

NOT_CONV, CONV = "#e8e8e8", "#2c5d8f"              # light grey -> deep blue
CMAP = LinearSegmentedColormap.from_list("conv", [NOT_CONV, CONV])

fig, axes = plt.subplots(1, len(masses), figsize=(4 * len(masses), 4),
                         sharey=True, squeeze=False, layout="constrained")

for ax, mpbh in zip(axes[0], masses):
    d = d_src[d_src.M_PBH == mpbh]
    Z = to_grid(d, "converged").to_numpy()

    ax.pcolormesh(X, Y, Z, cmap=CMAP, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True)
    if 0 < np.nanmean(Z) < 1:
        ax.contour(X, Y, Z, levels=[0.5], colors="k", linewidths=1.5)

    if log_rho_axis:
        ax.set_yscale("log")
    ax.set_title(rf"$M_{{\rm PBH}} = {mpbh:.1e}$ g")
    ax.set_xlabel(r"$\log_{10} N$")

axes[0, 0].set_ylabel(r"$\rho_0$ [g cm$^{-3}$]")
legend_items = [
    Patch(color=CONV, label="converged"),
    Patch(color=NOT_CONV, label="not converged"),
    Line2D([], [], color="k", lw=1.5, label="convergence boundary"),
]
fig.legend(handles=legend_items, loc="outside upper center", ncol=3, frameon=False)
fig.savefig("./img/SinglePBH_convergence_row.png", bbox_inches="tight")
plt.show()