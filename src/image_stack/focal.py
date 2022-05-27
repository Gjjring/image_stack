import numpy as np
from numpy import ma
from abc import ABC, abstractmethod
import scipy.optimize
import image_stack.utils as utils
import matplotlib.pyplot as plt
from image_stack.statistics import InformationCriteria, information_criterion

def create_model(model_name, plot):
    if model_name == 'gaussian':
        return GaussianModel(plot)
    if model_name == 'polynomial':
        return Polynomial(plot)
    else:
        raise ValueError("invalid model name: {}".format(model_name))


class FocalPlane():


    def __init__(self, z, data, model='gaussian', plot=False):
        self.z = z
        self.data = data
        self.model = create_model(model, plot)

    def fit(self, fit_window=None, fit_center=None):
        if fit_window is not None:
            data = ma.masked_where(np.abs(self.z-fit_center)>fit_window, self.data).compressed()
            z = ma.masked_where(np.abs(self.z-fit_center)>fit_window, self.z).compressed()
        else:
            data = self.data
            z = self.z
        return self.model.fit_parameters(z, data)

    def _evalulate_info_criterion(self, fit_window_width, criterion):
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
        z_range = (self.z[-1]-self.z[0])
        fit_window = z_range
        focal_pos0 = self.fit()[0]
        self.current_focus_estimate = focal_pos0
        criterion = InformationCriteria.BIC
        obj_fun = lambda x : self._evalulate_info_criterion(x, criterion)
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


class GaussianModel(ModelBase):

    def __init__(self, plot):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = 3

    def _init_params(self, z, data):
        a = np.max(data)
        b = z[np.where(np.isclose(data,a))[0][0]]
        z_range = (z[-1]-z[0])
        c = z_range*0.25
        self.parameters = [a, b, c]

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        self.__parameters['a'] = params[0]
        self.__parameters['b'] = params[1]
        self.__parameters['c'] = params[2]

    def evaluate(self, z):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return utils.gaussian([a, b, c], z)

    def get_bounds(self, z, data):
        z_range = z[-1]-z[0]
        bounds = []
        bounds.append((-np.inf, np.min(z), -np.inf))
        bounds.append((np.inf, np.max(z), z_range*0.5))
        return bounds

    def fit_parameters(self, z, data):
        self._init_params(z, data)
        start_vals = list(self.parameters.values())
        minimize_func = lambda x : self.residuals(x, z, data)
        bounds = self.get_bounds(z, data)
        res = scipy.optimize.least_squares(minimize_func,
                                            method='dogbox',
                                            loss='cauchy',
                                            x0=start_vals,
                                            bounds=bounds)

        self.parameters = res['x']
        if self.plot:
            self._plot_focal_function(z, self.evaluate(z))
        squared_residuals = np.sum(np.abs(np.power(self.residuals(res['x'], z, data), 2)))
        focal_plane_estimate = res['x'][1]
        return focal_plane_estimate, squared_residuals, z.size

class Polynomial(ModelBase):

    def __init__(self, plot, order=2):
        super().__init__(plot)
        self.__parameters = {}
        self.n_parameters = order

    def _init_params(self, z, data):
        coeffs = np.polyfit(z, data, self.n_parameters)
        #print(self.n_parameters, coeffs)
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

    def _evalulate_info_criterion(self, order, z, data, criterion):
        """
        evaluate the information criteria for the current model and fit window.

        Parameters
        ----------
        fit_window_width: float
            width of fitting window
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
        obj_fun = lambda x : self._evalulate_info_criterion(x, z, data,
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
