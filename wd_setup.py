import numpy as np

me = 9.1093837e-31
Mp = 1.67262192e-27

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled, gamma):
        self.Ye = Ye
        self.rhoc_scaled = rhoc_scaled

        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.k = Ye * me / Mp

        self.GAMMA = gamma

    #def _gamma(self, x):
    #    return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    
    def _x(self, rhob):
        return (rhob) ** (1/3)
    
    def get_derivative(self, state, rb):
        # state = [rho, drho/dr]
        rho = state[0]
        drhodr = state[1]

        d2rhodr2 =  - (2 / rb) * drhodr - (self.GAMMA - 2) * (drhodr ** 2) / rho - rho ** (3 - self.GAMMA)
        dmdr = rho * (rb ** 2)
        return np.array([drhodr, d2rhodr2, dmdr])