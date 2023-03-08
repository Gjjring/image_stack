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

def tophat(params, z):
    """
    Tophat function with parameters: center, height and width and zero level

    Parameters
    ----------
    params : list [4,] of floats
        The function parameters
    z : np.ndarray <N,>
        The function domain

    Returns
    -------
    the evaluated function values
    """
    center = params[0]
    height = params[1]
    width = params[2]
    zero_level = params[3]
    evaluated = np.ones(z.size)*zero_level
    indices = np.abs(z-center) < width*0.5
    evaluated[indices] += np.ones(z.size)[indices]*height
    return evaluated


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)
