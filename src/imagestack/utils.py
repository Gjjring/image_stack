import numpy as np
from collections.abc import Iterable

def gaussian(params, z):
    a = params[0]
    b = params[1]
    c = params[2]
    return a*np.exp( -(z-b)**2/(2*c**2))

def gaussian2D(params, x, y):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return a*np.exp( -1*((x-b)**2/(2*d**2) + (y-c)**2/(2*d**2)))

def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)
