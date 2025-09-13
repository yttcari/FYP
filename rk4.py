import numpy as np

# state = (rho, enclosed mass)

def rk4(derivative, dr, rb, state):
    k1 = derivative(rb=rb, state=state)
    k2 = derivative(rb=rb+dr/2, state=state+dr*k1*0.5)
    k3 = derivative(rb=rb+dr/2, state=state+dr*k2*0.5)
    k4 = derivative(rb=rb+dr, state=state+dr*k3)
    return state + (dr/6) * (k1 + 2 * k2 + 2 * k3 + k4)