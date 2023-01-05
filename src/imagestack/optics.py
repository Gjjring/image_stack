import numpy as np

def fresnel_t_s(n1, theta1, n2, theta2):
    """
    Calculate the complex amplitude transmission coefficient at a planar
    interface for s polarization.

    Parameters
    ----------
    n1: complex
        refractive index on incident side
    theta1: float
        incident angle in degrees
    n2: complex
        refractive index on outgoing side
    theta1: float
        outgoing angle in degrees
    """
    denom = n2*np.cos(np.radians(theta2)) + n1*np.cos(np.radians(theta1))
    return 2*n1*np.cos(np.radians(theta1))/denom

def fresnel_t_p(n1, theta1, n2, theta2):
    """
    Calculate the complex amplitude transmission coefficient at a planar
    interface for p polarization.

    Parameters
    ----------
    n1: complex
        refractive index on incident side
    theta1: float
        incident angle in degrees
    n2: complex
        refractive index on outgoing side
    theta1: float
        outgoing angle in degrees
    """
    denom = n2*np.cos(np.radians(theta1)) + n1*np.cos(np.radians(theta2))
    return 2*n1*np.cos(np.radians(theta1))/denom

def fresnel_r_s(n1, theta1, n2, theta2):
    """
    Calculate the complex amplitude reflection coefficient at a planar
    interface for s polarization.

    Parameters
    ----------
    n1: complex
        refractive index on incident side
    theta1: float
        incident angle in degrees
    n2: complex
        refractive index on non incident side
    """
    #theta2 = theta1 #law of reflection
    num =   n1*np.cos(np.radians(theta1)) - n2*np.cos(np.radians(theta2))
    denom = n1*np.cos(np.radians(theta1)) + n2*np.cos(np.radians(theta2))
    return num/denom

def fresnel_r_p(n1, theta1, n2, theta2):
    """
    Calculate the complex amplitude reflection coefficient at a planar
    interface for p polarization.

    Parameters
    ----------
    n1: complex
        refractive index on incident side
    theta1: float
        incident angle in degrees
    n2: complex
        refractive index on non incident side
    """
    #theta2 = theta1 #law of reflection
    num =   n2*np.cos(np.radians(theta1)) - n1*np.cos(np.radians(theta2))
    denom = n2*np.cos(np.radians(theta1)) + n1*np.cos(np.radians(theta2))
    return num/denom

def snell(n1, theta1, n2):
    """
    Calculate the propagation angle in region 2 at a material interface

    Parameters
    ----------
    n1: complex
        refractive index on incident side
    theta1: float
        incident angle in degrees
    n2: complex
        refractive index on non incident side
    """
    return np.degrees(np.real(np.arcsin(np.clip((n1/n2)*np.sin(np.radians(theta1)), -1., 1.))))
