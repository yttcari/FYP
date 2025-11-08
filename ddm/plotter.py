import numpy as np
import matplotlib.pyplot as plt

class plotter:
    def __init__(self, wd):
        self.wd = wd
        self.history = wd.history
        self.decay_dm = wd.decay_dm
        self._process_data()

    def _process_data(self):
        if self.decay_dm:
            # History columns: [rb, nm_rho, pdm_rho, ddm_rho, nm_M, pdm_M, ddm_M]
            self.rb = self.history[:, 0]
            self.nm_rho = self.wd.rhobar2rho(self.history[:, 1])
            self.pdm_rho = self.wd.rhobar2rho(self.history[:, 2])
            self.ddm_rho = self.wd.rhobar2rho(self.history[:, 3])
            
            self.nm_M = self.wd.mbar2m(self.history[:, 4])
            self.pdm_M = self.wd.mbar2m(self.history[:, 5])
            self.ddm_M = self.wd.mbar2m(self.history[:, 6])
            self.total_M = self.nm_M + self.pdm_M + self.ddm_M
        else:
            # History columns: [rb, nm_rho, nm_M]
            self.rb = self.history[:, 0]
            self.nm_rho = self.wd.rhobar2rho(self.history[:, 1])
            self.nm_M = self.wd.mbar2m(self.history[:, 2])
            self.total_M = self.nm_M # Total mass is just normal matter mass

        # Radius in km
        self.r_km = self.wd.rbar2r(self.rb)
        
    def plot_profiles(self, title="White Dwarf Structure Profile"):
        if self.decay_dm:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            fig.suptitle(title, fontsize=16)

            ax1 = axes[0][0]
            ax1.plot(self.r_km, self.nm_rho, label='Normal Matter (NM)', color='blue')
            #ax1.plot(self.r_km, self.pdm_rho, label='Parent Dark Matter (PDM)', color='red', linestyle='--')
            #ax1.plot(self.r_km, self.ddm_rho, label='Daughter Dark Matter (DDM)', color='green', linestyle=':')
            ax1.set_xlabel('Radius (km)')
            ax1.set_ylabel(r'Density $\rho$ (g/cm$^3$)')
            ax1.set_title('Density Profiles')
            ax1.legend()
            
            ax2 = axes[0][1]
            ax2.plot(self.r_km, self.nm_M, label='Normal Matter Mass', color='blue', linewidth=2)
            ax2.plot(self.r_km, self.pdm_M, label='PDM Mass', color='black', linestyle='-')
            #ax2.plot(self.r_km, self.pdm_M, label='PDM Mass', color='red', linestyle='--')
            ax2.plot(self.r_km, self.ddm_M, label='DDM Mass', color='black', linestyle=':')
            ax2.set_xlabel('Radius (km)')
            ax2.set_ylabel(r'Mass $M(r)$ ($M_{\odot}$)')
            ax2.set_title(f'Enclosed Mass Profile (NM $M_{{NM}} = {self.nm_M[-1]:.3f} M_{{\odot}}$)')
            ax2.legend()

            ax3 = axes[1][0]
            ax3.plot(self.r_km, self.pdm_rho, label='PDM density', color='black', linewidth=2)
            ax3.set_xlabel('Radius (km)')
            ax3.set_ylabel(r'$\rho_{\odot}$')
            ax3.set_title(r'$\rho_{{PDM}} (g/cc$)')
            ax3.legend()

            ax4 = axes[1][1]
            ax4.plot(self.r_km, self.ddm_rho, label='DDM density', color='black', linewidth=2)
            ax4.set_xlabel('Radius (km)')
            ax4.set_ylabel(r'$\rho_{\odot}$')
            ax4.set_title(r'$\rho_{{DDM}} (g/cc$)')
            ax4.legend()
            
        else:
            # Case without Dark Matter 
            fig, axes = plt.subplots(1, 2, figsize=(8,4))
            fig.suptitle(title + " (No Dark Matter)", fontsize=16)
            
            ax1 = axes[0]
            ax1.plot(self.r_km, self.nm_rho, label='Normal Matter (NM)', color='blue')
            ax1.set_xlabel('Radius (km)')
            ax1.set_ylabel(r'Density $\rho$ (g/cm$^3$)')
            ax1.set_title('Density Profile')
            
            ax2 = axes[1]
            ax2.plot(self.r_km, self.total_M, label='Total Enclosed Mass', color='black', linewidth=2)
            ax2.set_xlabel('Radius (km)')
            ax2.set_ylabel(r'Mass $M(r)$ ($M_{\odot}$)')
            ax2.set_title(f'Enclosed Mass Profile (Total $M_{{WD}} = {self.total_M[-1]:.3f} M_{{\odot}}$)')
            ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()