import numpy as np

def gaussian(params, z):
    a = params[0]
    b = params[1]
    c = params[2]
    return a*np.exp( -(z-b)**2/(2*c**2))
