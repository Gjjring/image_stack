import numpy as np
from numpy import ma
from abc import ABC, abstractmethod
import scipy.optimize
import imagestack.utils as utils
import matplotlib.pyplot as plt
from imagestack.statistics import InformationCriteria, information_criterion

def create_model(model_name, plot):
    if model_name == 'gaussian':
        return GaussianModel(plot)
    if model_name == 'polynomial':
        return Polynomial(plot)
    else:
        raise ValueError("invalid model name: {}".format(model_name))

def _mask_z_and_data(window, center, z, data):
    data = ma.masked_where(np.abs(z-center)>window, data).compressed()
    z = ma.masked_where(np.abs(z-center)>window, z).compressed()
    return data, z

class FocalPlane():

    def __init__(self, z, data, model='gaussian', plot=False):
        self.z = z
        self.data = data
        self.model = create_model(model, plot)

    def fit(self, fit_window=None, fit_center=None):
        if fit_window is not None:
            data, z = _mask_z_and_data(fit_window, fit_center,
                                            self.z, self.data)
        else:
            data = self.data
            z = self.z
        return self.model.fit_parameters(z, data)

    def _evaluate_info_criterion(self, fit_window_width, criterion):
        """
        evaluate the information criteria for the current model and fit window.

        Parameters
        ----------
        fit_window_width: float
            width of fitting window
        criterion: InformationCriteria Enum
            the information criteria to be minimized
        """
        fit_output = self.fit(fit_window=fit_window_width,
                              fit_center=self.current_focus_estimate)
        n_params = self.model.n_parameters
        residual = fit_output[1]
        n_data_points = fit_output[2]
        return information_criterion(residual, n_params,
                                     n_data_points, criterion)


    def optimize_fit_window(self, iterations=1):
        """
        Find the data range for the current model to maximise goodness of fit

        Assuming we have chosen a model which can potentially fit our data, we
        seek the width of a window function which will exclude data that does
        not fit to the model. By employing an information criterion, we ensure
        a balance between throwing out outliers and keep as much data as
        possible inside the window.

        Parameters
        ----------
        iterations: int
            number of times to iterate routine

        Returns
        -------
        float
            The size of the fitting window in z coordinates
        """
        z_range = (np.max(self.z)-np.min(self.z))
        fit_window = z_range
        focal_pos0 = self.fit()[0]
        self.current_focus_estimate = focal_pos0
        criterion = InformationCriteria.BIC
        obj_fun = lambda x : self._evaluate_info_criterion(x, criterion)
        min_fit_window_width = z_range*0.2
        max_fit_window_width = z_range
        for iteration in range(iterations):
            res = scipy.optimize.minimize_scalar(obj_fun,
                                           bounds=(min_fit_window_width,
                                                   max_fit_window_width),
                                           method='bounded',
                                           options={'maxiter':40,
                                                    'disp':False})

            optimal_fit_window_width = res['x']
            if iterations > 1:
                fit_output = self.fit(fit_window=res['x'],
                                      fit_center=self.current_focus_estimate)
                self.current_focus_estimate = fit_output[0]
        return optimal_fit_window_width

class ModelBase(ABC):

    def __init__(self, plot):
        self.plot = plot
        pass

    def n_parameters(self):
        return self.n_parameters

    @abstractmethod
    def evaluate(self, z):
        pass

    @abstractmethod
    def _init_params(self, z, data):
        pass

    def residuals(self, model_params, z, data):
        self.parameters = model_params
        residuals = (self.evaluate(z)-data)
        return residuals

    def _plot_focal_function(self, z, data):
        label_str = ""
        for key, val in self.parameters.items():
            label_str += "{}:{:.2e} ".format(key, val)
            if len(label_str) > 47:
                label_str += "..."
                break
        plt.plot(z, data, label=label_str)

    def _plot_focal_function_2D(self, x, y, data):
        label_str = ""
        for key, val in self.parameters.items():
            label_str += "{}:{:.2e} ".format(key, val)
            if len(label_str) > 47:
                label_str += "..."
                break
        plt.pcolormesh(x, y, data, label=label_str, shading='gouraud')


class GaussianModel(ModelBase):

    def __init__(self, plot):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = 4

    def _init_params(self, z, data):
        a = np.max(data)-np.min(data)
        b = z[np.where(np.isclose(data, np.max(data)))[0][0]]
        z_range = np.max(z)-np.min(z)
        c = z_range*0.5
        d = np.min(data)
        self.parameters = [a, b, c, d]

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        self.__parameters['a'] = params[0]
        self.__parameters['b'] = params[1]
        self.__parameters['c'] = params[2]
        self.__parameters['d'] = params[3]

    def evaluate(self, z):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']
        return utils.gaussian([a, b, c], z) + d

    def get_bounds(self, z, data):
        z_range = np.max(z)-np.min(z)
        bounds = []
        bounds.append((0., np.min(z), 0., -np.max(data)))
        bounds.append((np.inf, np.max(z), z_range*2, np.max(data)))
        return bounds

    def fit_parameters(self, z, data, init_params=None):
        if init_params is None:
            self._init_params(z, data)
        else:
            self.parameters = init_params
        start_vals = list(self.parameters.values())
        minimize_func = lambda x : self.residuals(x, z, data)
        bounds = self.get_bounds(z, data)
        res = scipy.optimize.least_squares(minimize_func,
                                            method='trf',
                                            loss='linear',
                                            x0=start_vals,
                                            bounds=bounds)

        self.parameters = res['x']
        if self.plot:
            self._plot_focal_function(z, self.evaluate(z))
        squared_residuals = np.sum(np.abs(np.power(self.residuals(res['x'], z, data), 2)))
        focal_plane_estimate = res['x'][1]
        return focal_plane_estimate, squared_residuals, z.size

class GaussianModel2D(ModelBase):

    def __init__(self, plot):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = 5

    def _init_params(self, x, y, data):
        a = np.amax(data)-np.amin(data)
        indices = np.where(np.isclose(data, np.amax(data)))
        b = x[np.where(np.isclose(data, np.amax(data)))[0], 0][0]
        c = y[0, np.where(np.isclose(data, np.amax(data)))[1]][0]
        x_range = np.amax(x)-np.amin(x)
        d = x_range*0.5
        e = np.amin(data)
        self.parameters = [a, b, c, d, e]

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        self.__parameters['a'] = params[0]
        self.__parameters['b'] = params[1]
        self.__parameters['c'] = params[2]
        self.__parameters['d'] = params[3]
        self.__parameters['e'] = params[4]

    def evaluate(self, x, y):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']
        e = self.parameters['e']
        return utils.gaussian2D([a, b, c, d], x, y) + e

    def get_bounds(self, x, y, data):
        x_range = np.amax(x)-np.amin(x)
        y_range = np.amax(y)-np.amin(y)
        bounds = []
        bounds.append((0., np.amin(x), np.amin(y), 0., -np.amax(data)))
        bounds.append((np.inf, np.amax(x), np.amax(y), x_range*2, np.amax(data)))
        return bounds

    def residuals(self, model_params, x, y, data):
        self.parameters = model_params
        residuals = (self.evaluate(x, y)-data).flatten()
        return residuals


    def fit_parameters(self, x, y, data, init_params=None):
        if init_params is None:
            self._init_params(x, y, data)
        else:
            self.parameters = init_params
        start_vals = list(self.parameters.values())
        minimize_func = lambda t : self.residuals(t, x, y, data)
        bounds = self.get_bounds(x, y, data)
        res = scipy.optimize.least_squares(minimize_func,
                                            method='trf',
                                            loss='linear',
                                            x0=start_vals,
                                            bounds=bounds)

        self.parameters = res['x']
        if self.plot:
            self._plot_focal_function_2D(x, y, self.evaluate(x, y))
        squared_residuals = np.sum(np.abs(np.power(self.residuals(res['x'],
                                                                  x, y, data), 2)))
        center_x = res['x'][1]
        center_y = res['x'][2]
        return center_x, center_y, squared_residuals, x.shape


class CenteredGaussianModel(GaussianModel):

    def __init__(self, plot):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = 3

    def _init_params(self, z, data):
        a = np.max(data)-np.min(data)
        z_range = np.max(z)-np.min(z)
        c = z_range*0.05
        d = np.min(data)
        self.parameters = [a, c, d]

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        self.__parameters['a'] = params[0]
        self.__parameters['c'] = params[1]
        self.__parameters['d'] = params[2]

    def evaluate(self, z):
        a = self.parameters['a']
        c = self.parameters['c']
        d = self.parameters['d']
        return utils.gaussian([a, 0., c], z) + d

    def get_bounds(self, z, data):
        z_range = np.max(z)-np.min(z)
        bounds = []
        bounds.append((0., 0., -np.max(data)))
        bounds.append((np.inf, z_range*2, np.max(data)))
        return bounds


class Polynomial(ModelBase):

    def __init__(self, plot, order=2):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = order

    def _init_params(self, z, data):
        coeffs = np.polyfit(z, data, self.n_parameters)
        self.parameters = coeffs

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        self.__parameters = {}
        for ii, param in enumerate(params[::-1]):
            self.__parameters['c{}'.format(ii)] = param

    def evaluate(self, z):
        data = np.zeros(z.shape)
        for tag, coeff in self.parameters.items():
            power = int(tag[1:])
            data += coeff*np.power(z, power)

        return data

    def _evaluate_info_criterion(self, order, z, data, criterion):
        """
        evaluate the information criteria for the current model and fit window.

        Parameters
        ----------
        order: int or float
            number of free model parameters
        z: np.ndarray of floats
            the z positions
        data: np.ndarray of floats
            the data evaluated at the z positions
        criterion: InformationCriteria Enum
            the information criteria to be minimized
        """
        self.n_parameters = int(order)
        self._init_params(z, data)
        model_data = self.evaluate(z)
        residual = np.sum(np.power(data-model_data,2))
        n_params = self.n_parameters
        n_data_points = z.size
        return information_criterion(residual, n_params,
                                     n_data_points, criterion)

    def fit_parameters(self, z, data):
        self._init_params(z, data)
        criterion = InformationCriteria.BIC
        obj_fun = lambda x : self._evaluate_info_criterion(x, z, data,
                                                            criterion)
        min_order = 2
        max_order = 12
        res = scipy.optimize.minimize_scalar(obj_fun,
                                       bounds=(min_order,
                                               max_order),
                                       method='bounded',
                                       options={'maxiter':40,
                                                'disp':False})

        self.n_parameters = int(res['x'])
        self._init_params(z, data)
        model_data = self.evaluate(z)
        if self.plot:
            self._plot_focal_function(z, self.evaluate(z))
        squared_residuals = np.sum(np.abs(np.power(data-model_data, 2)))

        xtol_estimate = np.abs(z[1]-z[0])*1e-5
        obj_fun2 = lambda x : -self.evaluate(x)
        res2 = scipy.optimize.minimize_scalar(obj_fun2,
                                              bounds=(np.min(z), np.max(z)),
                                              options={'disp':False,
                                                        'xatol':xtol_estimate},
                                              method='bounded')
        focal_plane_estimate = res2['x']

        return focal_plane_estimate, squared_residuals, z.size
