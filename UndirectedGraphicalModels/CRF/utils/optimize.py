import numpy as np
from scipy.optimize import minimize



def rosen(x):
    val=(1-x[0])**2+ 100*(x[1]-x[0]**2)**2
    return val

def rosen_der(x):
    der = np.zeros_like(x)
    der[0]=400*x[0]**3 -400*x[0]*x[1] + 2*x[0]-2
    der[1]=-200*x[0]**2 + 200*x[1]
    return der


