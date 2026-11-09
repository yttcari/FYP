import numpy as np
from WhiteDwarf.constants import *
import os
import pandas as pd
from scipy.special import erf

hbar = h / (2 * np.pi)

class proton_decay:
	def __init__(self, k, **kwargs):
		self.k = k # s-1
	
	def luminosity(self, r, m, **kwargs):
		"""
		m: enclosed mass in g
		k: decay constant in 1/s
		E_gamma: gamma ray energy in erg
		mp: proton mass in g
		"""
		L = m * self.k * proton_energy/ mp # unit: erg/s
		return L

class No_source:
	def __init__(self, **kwargs):
		self.source_type = 'No source'

	def get_mass(self):
		return 0
	
	def luminosity(self, **kwargs):
		return 0

class BasePBH:
	source_type = None

	def get_spectra(self, spectra_type):
		mass = f"{self.pbhM:.0e}"
		spectra_dir = f"./blackhawk_v2.3/results/Hazma/M{mass}/instantaneous_{spectra_type}_spectra.txt"

		if not os.path.exists(spectra_dir):
			print(f"The entered PBH mass has no {spectra_type} spectra computed. Please compute it using BlackHawk before you run this. TERMINATE")
			raise ValueError(f"No {spectra_type} Spectra found")

		return pd.read_csv(spectra_dir, sep=r'\s+', skiprows=1)

	def luminosity(self, E_fermi, E_el_int, **kwargs):
		L = 0.
		pri_df = self.get_spectra('primary')
		sec_df = self.get_spectra('secondary')

		E_pri = pri_df['energy/particle'].values
		
		no_anti_count = ['electron', 'photon', 'gluons', 'higgs', 'Z0',]
		just_ignore = ['graviton', 'DM', 'neutrinos',]


		for sp in pri_df.columns[1:]:
			# include anti_particle count
			if sp not in no_anti_count and sp not in just_ignore: # these species do not have anti particle
				dN_dE = pri_df[sp].values
				m = rest_mass[sp]
				
				E_dep = 2.*E_pri - m
				L += np.trapezoid(dN_dE * E_dep, E_pri)

			# ignore photon as it's included in sec spectrum
			# ignore electron for annhilation
			# ignore neutrino assuming it's transparent
			elif sp != 'photon' and sp != 'electron' and sp not in just_ignore:
				dN_dE = pri_df[sp].values
				m = rest_mass[sp]
				L += np.trapezoid(dN_dE * E_pri, E_pri)

		# include photon from pri spectrum and hadronization
		E = sec_df['energy/particle'].values

		dN_dE_photon = sec_df['photon'].values
		L += np.trapezoid(dN_dE_photon * E, E)

		# electronic contribution
		dN_dE_e = 0.5 * sec_df['electron'].values
		E_kinetic = E - rest_mass['electron'] 

		# pauli blocking
		mask = E_kinetic >= E_fermi
		if mask.sum() >= 2:
			E_dep_electron = E_kinetic[mask] - E_fermi
			L += np.trapezoid(dN_dE_e[mask] * E_dep_electron, E[mask])

		# annhilation energy
		E_dep_positron = E + rest_mass['electron'] + E_el_int
		L += np.trapezoid(dN_dE_e * E_dep_positron, E)

		N = self._N_enclosed(kwargs['r']) if hasattr(self, '_N_enclosed') else self.N
		return L * GEV_TO_ERG * N

	def get_mass(self, r=None):
		raise NotImplementedError


class singlePBH(BasePBH):
	source_type = 'SinglePBH'

	def __init__(self, pbhM, N=1, **kwargs):
		self.pbhM = pbhM
		self.N = N
		print(f"Hawking temperature of the PBH is {self.hawking_T():.3e}")

	def hawking_T(self):
		return hbar * c ** 3 / (8 * np.pi * G * self.pbhM * kB)

	def get_mass(self):
		return self.pbhM * self.N


class uniformPBH(BasePBH):
	source_type = 'UniformPBH'

	def __init__(self, pbhM, N, R, **kwargs):
		self.pbhM = pbhM
		self.N = N
		self.R = R

	def _N_enclosed(self, r):
		return self.N * min(r / self.R, 1) ** 3

	def get_mass(self, r):
		return self.pbhM * self.N * min(r / self.R, 1) ** 3


class gaussianPBH(BasePBH):
	source_type = 'GaussianPBH'

	def __init__(self, pbhM, N, sigma, **kwargs):
		# N: total number of PBHs (integrated to r -> infinity)
		self.pbhM = pbhM
		self.N = N
		self.sigma = sigma
		self.N0 = N / (4 * np.pi * self.sigma**3 * np.sqrt(np.pi / 2))

	def _N_enclosed(self, r):
		r = np.asarray(r)
		u = r / (np.sqrt(2) * self.sigma)
		return 4 * np.pi * self.N0 * self.sigma**3 * (
			np.sqrt(np.pi / 2) * erf(u) - (r / self.sigma) * np.exp(-u**2)
		)

	def get_mass(self, r):
		return self.pbhM * self._N_enclosed(r)