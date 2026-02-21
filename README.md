# Introduction
The main goal of this project is to analyze how different energy source affect the white dwarf both hydrodynamically and thermally.

All the units are in cgs unit, excepts for the hydro integration within white dwarf.

# Energy Source Implemented
- Proton decay with arbitrary decay constant $k$
- Primordial black hole (PBH) that emits Hawking radiation (in progress)

The hydro and thermal integration are integrated alternatively until the parameters converge.

Sample usage for initial iteration:
```
wd = WhiteDwarf(Ye=0.5, rhoc_scaled=rhoc_scaled, source=source.No_source, Z=6)
wd.hydro_integrate()

updated_source = source_function(rho_min=np.min(wd.rho_profile), rho_max=np.max(wd.rho_profile), Tmin=0, Tmax=0, M_WD=wd.M_profile[-1]*wd.M0, eos=wd.table, atomic_num=wd.Z, mass_num=wd.A)
wd.source = updated_source

wd.thermo_integrate()
```