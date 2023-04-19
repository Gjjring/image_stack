import numpy as np
from numpy import ma
from enum import Enum

class InformationCriteria(Enum):
    BIC = 0
    AIC = 1
    HQC = 2


def information_criterion(residual, n_params, n_data_points, criterion):
    """
    evalute the information criterion

    Parameters
    ----------


    """
    reduced_chi_squared = residual/(n_data_points-n_params)
    residual = reduced_chi_squared
    if criterion == InformationCriteria.BIC:
        return bic(residual, n_params, n_data_points)
    elif criterion == InformationCriteria.AIC:
        return aic(residual, n_params, n_data_points)
    elif criterion == InformationCriteria.HQC:
        return hqc(residual, n_params, n_data_points)
    else:
        raise ValueError("unknown criterion: {}".format(criterion))


def bic(residual, n_params, n_data_points):
    return (np.log(n_data_points)*n_params +
            n_data_points*np.log(residual))

def aic(residual, n_params, n_data_points):
    return (2*n_params + n_data_points*np.log(residual))

def hqc(residual, n_params, n_data_points):
    return (2*n_params*np.log(np.log(n_data_points)) +
            n_data_points*np.log(residual))

def residuals(data1, data2):
    """
    statistical data comparing the pixels.

    Parameters
    ----------
    data1: (N,) np.ndarray
        pixel data1
    data2: (N,) np.ndarray
        pixel data2
    """
    return_data = {}
    return_data['rms_dif'] = np.sqrt(np.mean((data1-data2)**2))
    return_data['max_dif'] = np.max(np.abs(data1-data2))
    return_data['squared_dif'] = np.sum((data1-data2)**2)
    return_data['cubic_dif'] = np.sum((np.abs(data1-data2))**3)
    #ls_dif_norm = 0.
    #for i_ns in range(self.n_layers):
    #    ls_dif_norm += np.sum((data1[:, :, i_ns]-data2[:, :, i_ns])**2) / np.mean(data1[:,:, i_ns])**2
    #return_data['ls_dif_norm'] = ls_dif_norm
    return return_data
