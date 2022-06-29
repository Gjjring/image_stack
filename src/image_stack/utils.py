import numpy as np
from collections.abc import Iterable

def gaussian(params, z):
    a = params[0]
    b = params[1]
    c = params[2]
    return a*np.exp( -(z-b)**2/(2*c**2))

def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)
