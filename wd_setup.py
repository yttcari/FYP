import numpy as np

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled):
        self.Ye = Ye
        self.rhoc_scaled = rhoc_scaled

        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 

    def _gamma(self, x):
        return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    
    def _x(self, rhob):
        return (rhob) ** (1/3)
    
    def get_derivative(self, state, rb):
        rhob = state[0]
        mb = state[1]

        x = self._x(rhob)
        gamma = self._gamma(x)

        dmdr = rhob * (rb ** 2)
        drhodr = -(mb * rhob) / (gamma * (rb ** 2))

        return np.array([drhodr, dmdr])