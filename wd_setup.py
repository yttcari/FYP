import numpy as np

me = 9.1093837e-31 # kg
mp = 1.67262192e-27 # kg
G = 6.67430e-8 # cgs

class WhiteDwarf:
    def __init__(self, Ye, rhoc_scaled):
        self.Ye = Ye
        self.rhoc_scaled = rhoc_scaled

        self.rho0 = 9.79e5 / Ye
        self.R0 = 7.72e8 * Ye
        self.M0 = 5.67e33 * (Ye ** 2) 
        self.GAMMA = 5/3
        self.n0 = 5.89e29

    def _gamma(self, x):
        return (x ** 2) / (3 * np.sqrt(1 + (x ** 2)))
    
    def _x(self, rhob):
        return (rhob) ** (1/3)

    def get_derivative(self, state, rb): 
        # state = [rho, m]
        rho, m = np.maximum(state[0], np.finfo(np.float32).eps), np.maximum(state[1], np.finfo(np.float32).eps)

        x = self._x(rho)
        gamma = self._gamma(x)

        P = self.Ye * me * (x * np.sqrt(x ** 2 + 1) * (2 * x ** 2 -3) + 3 * np.arctanh(x / np.sqrt(x ** 2 + 1))) / (8 * mp) / self.rho0

        dmdr = rb**2 * rho 

        if rb < 1e-4:
            #m = (1/3) * rho * (rb**3)
            t0 = rb * rho ** 2 / 3
            t1 = (1 + P/rho)
            t2 = (1 + P/ (rho /3))
            t3 = 1 / (1 - self.Ye * me * (2/3) * rho * rb ** 2 / mp)
        else:
            t0 = (m * rho / rb**2)
            t1 = (1 + P/rho)
            t2 = (1 + (rb**3 * P)/m)
            t3 = 1 / (1 - (2 * m * self.Ye * me / mp)/rb)

        dPdr = -t0 * t1 * t2 * t3
        #print(t1, t2, t3)

        drhodr = dPdr / gamma

        return np.array([drhodr, dmdr])