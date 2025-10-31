import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis 

#CONSTANT (CGS)
m_p = 1.6727e-24 
m_e = 9.1094e-28
c = 2.9979e10 
G = 6.6743e-8 
solar_mass = 1.9885e33
hbar = 1.0546e-27 
n_0 = (m_e**3) * (c/hbar)**3 / (3 * np.pi**2)

#PARAMETERS(CGS)
Y_e = 0.5
rho_c_NM = 1e8 
rho_c_DM = 1e7 
m_DM_particle_mass = 1.78e-25 
#k 
#l

#RESCALING (CGS)
rho_0 = 9.79e5 / Y_e  
R_0 = 7.72e8 * Y_e  # cm
M_0 = 5.67e33 * Y_e**2  # g

#DIFFERENTIAL EQUATIONS (DIMENSIONLESS)
constant_1  = (R_0 * m_p) / (rho_0 * Y_e * m_e * c**2)
constant_2 = (G * M_0 * rho_0)/(R_0**2)
constant_3 = ((m_p * m_e**3)/(Y_e * m_DM_particle_mass**4))**(1/3)
constant_4 = (Y_e * m_e) / m_p

def equations(r, state):
    M_NM, M_DM, rho_NM, rho_DM = state

    #SAFETY GUARDS
    rho_NM = max(rho_NM, 1e-20)
    rho_DM = max(rho_DM, 1e-20)

    #GAMMA
    x_NM = rho_NM**(1/3)
    x_DM = constant_3 * rho_DM**(1/3)
    gamma_NM = x_NM**2 / (3 * (1 + x_NM)**0.5)
    gamma_DM = x_DM**2 / (3 * (1 + x_DM)**0.5)

    #EPSILON PRIME
    def epsilon_prime_computer(x):
        if x < 1e-10: 
            return 0
        elif x < 1e-5: 
            return 2 * x
        else: 
            sqrt = np.sqrt(x**2 + 1)
            numerator = sqrt * np.log(sqrt + x) + 6 * x**5 + 5 * x**3 - x 
            denominator = 8 * x**2 * sqrt 
            return 3 * numerator / denominator
    
    eps_prime_NM = epsilon_prime_computer(x_NM)
    eps_prime_DM = epsilon_prime_computer(x_DM)

    #PRESSURES (PHYSICAL)
    P_NM = (1/3) * n_0 * m_e * (x_NM)**4 * eps_prime_NM * c**2
    P_DM = (1/3) * constant_3**(-3) * rho_0 * x_DM**4 * eps_prime_DM * c**2
    P_tot = P_NM + P_DM

    #TOV
    M_tot = M_NM + M_DM 
    term_2 = 1 + (4 * np.pi * r**3 * P_tot)/(M_tot * M_0 * c**2)
    term_3 = (1 - (2 * G * M_tot * M_0)/(r * R_0 * c**2))**(-1)

    TOV_NM = (1 + (P_NM/(rho_NM * rho_0 * c**2))) * term_2 * term_3 
    TOV_DM = (1 + (P_DM/(rho_DM * rho_0 * c**2))) * term_2 * term_3 

    #EQUATIONS 
    dM_NM_dr = r**2 * rho_NM 
    dM_DM_dr = r**2 * rho_DM 
    drho_NM_dr = - constant_1 / gamma_NM * (constant_2 * (M_NM + M_DM)/r**2 * rho_NM) * TOV_NM
    drho_DM_dr = - constant_4 * (M_NM + M_DM)/(gamma_DM * r**2) * rho_DM * TOV_DM

    return np.array([dM_NM_dr, dM_DM_dr, drho_NM_dr, drho_DM_dr])

#RK4 STEPPING
def step_computer(r, state, h): 
    k1 = equations(r, state)
    k2 = equations(r + h/2, state + k1 * h/2)
    k3 = equations(r + h/2, state + k2 * h/2)
    k4 = equations(r + h, state + k3 * h)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * h/6 #NEW STATE


#RK4 INTEGRATION 
def integration_computer(rho_c_NM, rho_c_DM, h=1e-5):
    r_0 = 1e-5
    m_NM_0 = (4 * np.pi * r_0**3 / 3) * rho_c_NM 
    m_DM_0 = (4 * np.pi * r_0**3 / 3) * rho_c_DM 
    state = np.array ([m_NM_0, m_DM_0, rho_c_NM, rho_c_DM])

    r_storage = [r_0]
    state_storage = [state.copy()]

    r = r_0

    while True: 
        state = step_computer(r, state, h)
        r += h

        state[2] = max(state[2], 0)
        state[3] = max(state[3], 0)

        r_storage.append(r)
        state_storage.append(state)

        if (state[2] < 1e-10 and state[3] < 1e-10) or r > 1.5: 
            break 
    
    r_storage = np.array(r_storage)
    state_storage = np.array(state_storage)

    return r_storage, state_storage[:, 0], state_storage[:, 1], state_storage[:, 2], state_storage[:, 3]

#RUN INTEGRATION
rho_c_NM_dimensionless = rho_c_NM / rho_0 
rho_c_DM_dimensionless = rho_c_DM / rho_0
r_storage, M_NM_storage, M_DM_storage, rho_NM_storage, rho_DM_storage = integration_computer(rho_c_NM_dimensionless, rho_c_DM_dimensionless, h=1e-4)
r_km = r_storage * R_0 * 1e-5
M_NM_solar = M_NM_storage * M_0 / solar_mass
M_DM_solar = M_DM_storage * M_0 / solar_mass
rho_NM_gcc = rho_NM_storage * rho_0 
rho_DM_gcc = rho_DM_storage * rho_0 

#SUMMARY 
print(f"Final radius: {r_km[-1]:.2f} km")
print(f"NM mass: {M_NM_solar[-1]:.4f} M_sun")
print(f"DM mass: {M_DM_solar[-1]:.4f} M_sun")
print(f"Total mass: {M_NM_solar[-1] + M_DM_solar[-1]:.4f} M_sun")

#PLOTTING
plt.figure(figsize=(12,8)) 

plt.subplot(2, 1, 1)
plt.plot(r_km, rho_NM_gcc)
plt.title("NM Density Profile")
plt.ylabel("Density (g/cc)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(r_km, rho_DM_gcc)
plt.title("DM Density Profile")
plt.ylabel("Density (g/cc)")
plt.xlabel("Radial distance (km)")
plt.grid(True)

plt.tight_layout()
plt.show()