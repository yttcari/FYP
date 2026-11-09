import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path.cwd().parent          # PBH_WD/

PBH_DIR = ROOT / "result" / "SinglePBH"     # PBH_M*/N*/rho0*.csv
NOSRC_DIR = ROOT / "result" / "No_source"   # rho0*.csv (flat)

PAT_PBH = re.compile(r"PBH_M([0-9.e+-]+)/N([0-9.e+-]+)/rho0([0-9.e+-]+)\.csv$")
PAT_RHO = re.compile(r"rho0([0-9.e+-]+)\.csv$")


def summarise(f):
    df = pd.read_csv(f)
    last = df.iloc[-1]
    ok = len(df) > 1 and np.isfinite(last[["radius", "mass"]]).all()
    return dict(R=last["radius"], M=last["mass"], T_c=df["temperature"].iloc[0] if "temperature" in df else np.nan, 
                n_steps=len(df), ok=ok, path=str(f))


# read SinglePBH runs
def load_pbh(d):
    rows = []
    for f in sorted(d.glob("PBH_M*/N*/rho0*.csv")):
        m = PAT_PBH.search(f.as_posix())
        if m:
            rows.append(dict(M_PBH=float(m[1]), N=float(m[2]), rho0=float(m[3]), **summarise(f)))
    return pd.DataFrame(rows)


# run WD without energy source
def load_nosource(d):
    rows = []
    for f in sorted(d.glob("rho0*.csv")):
        m = PAT_RHO.search(f.name)
        if m:
            rows.append(dict(rho0=float(m[1]), **summarise(f)))
    return pd.DataFrame(rows)


pbh = load_pbh(PBH_DIR)
nosrc = load_nosource(NOSRC_DIR)

for name, df in [("SinglePBH", pbh), ("No source", nosrc)]:
    if df.empty:
        print(f"{name}: no files found")
    else:
        print(f"{name}: {len(df)} runs, {int(df.ok.sum())} converged")
        bad = df[~df.ok]
        if len(bad):
            print(f"  {len(bad)} failed, e.g.:")
            print(bad.drop(columns=["path"]).head(5).to_string(index=False))

if pbh.empty or not pbh.ok.any():
    raise RuntimeError(f"No converged SinglePBH runs in {PBH_DIR}")

good = pbh[pbh.ok].sort_values(["M_PBH", "N", "rho0"])
ref = nosrc[nosrc.ok].sort_values("rho0") if not nosrc.empty else nosrc


# plotting
EXCLUDE = [3e14]

def plot_mr(cmap="viridis"):
    masses = [m for m in sorted(good.M_PBH.unique())
              if not np.any(np.isclose(m, EXCLUDE, rtol=1e-6))]
    norm = LogNorm(vmin=good.N.min(), vmax=good.N.max())

    fig, axes = plt.subplots(1, len(masses), figsize=(4.2 * len(masses), 4),
                             sharey=True, squeeze=False)
    for ax, mpbh in zip(axes[0], masses):
        for N, g in good[good.M_PBH == mpbh].groupby("N"):
            ax.plot(g.R, g.M, "o-", lw=1.5, ms=4, color=plt.get_cmap(cmap)(norm(N)))
        if not ref.empty:
            ax.plot(ref.R, ref.M, "k--", lw=1.5, label="No source", zorder=5)
        ax.set_title(rf"$M_{{\rm PBH}} = {mpbh:.1e}$ g")
        ax.set_xlabel("R [km]")
        ax.grid(alpha=0.3, which="both")

    axes[0, 0].set_ylabel(r"M [$M_\odot$]")
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(),
                 label="N (number of PBHs)")
    fig.suptitle("SinglePBH")
    return fig


fig = plot_mr()
Path("./img").mkdir(exist_ok=True)
fig.savefig("./img/MR_relation.png", dpi=200, bbox_inches="tight")
plt.show()