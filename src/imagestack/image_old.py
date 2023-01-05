import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import scipy.optimize
import scipy.io
import scipy.constants
import os
import optics
import utils
import basis_functions
from image_base import ImageBase
import pickle
from integrator import Integrator2D



class Image(ImageBase):

    """
    2D pixel image with z position

    Attributes
    ---------
    data: (N,M) np.ndarray
        The pixel data
    masked_array: (N,M) np.ma.maskedarray
        The masked pixel data
    x: (N,) np.ndarray
        The positions of the pixels in the x direction
    y: (M,) np.ndarray
        The positions of the pixels in the y direction
    z: float
        The z position of the image
    mask_shape: str
        The shape of the mask for data parameter fitting
    mask_region: str
        The region of the mask for data parameter fitting
    mask_constraint: float or (2,) tuple
        constraint for the mask for data parameter fitting
    label: str
        label used for plotting
    """

    def __init__(self, data, x, y, z):
        """
        Parameters
        ----------
        data: (N,M,P) np.ndarray
            The pixel data
        x: (N,) np.ndarray
            The positions of the pixels in the x direction
        y: (M,) np.ndarray
            The positions of the pixels in the y direction
        z: float
            The position of the image in the z direction
        """
        self.data = data
        self.masked_data = ma.array(data, copy=False)
        self.x = x
        self.y = y
        self.z = z

        self.mask_shape = None
        self.mask_region = None
        self.mask_constraint = 1.0

        self.label=""

    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        return np.meshgrid(self.x, self.y, indexing='ij')

    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        X, Y = self.get_cart_dimensions()
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        return R, PHI

    @classmethod
    def from_basis(image, basis_func, n, X, Y, polar=False):
        """
        returns a line image of a basis function of order n evaluated on x

        Parameters
        ----------
        basis_func: function handle
            the basis function to use
        n: int
            the order of the basis function
        X: nd.array<float>(N,M)
            the x positions over which to evaluate the objective function
        Y: nd.array<float>(N,M)
            the y positions over which to evaluate the objective function
        polar: bool
            use a polar coordinates
        """
        data = np.zeros((X.shape[0], Y.shape[1]))
        mode_range = range(mode_start, n_modes+mode_start)
        if polar:
            R = np.sqrt(X**2 + Y**2)
            PHI = np.arctan2(Y,X)
            dimension1 = R
            dimension2 = PHI
        else:
            dimension1 = X
            dimension2 = Y
        len_x = X.shape[0]
        len_y = Y.shape[1]
        for imode, mode in enumerate(mode_range):
            modal_func = basis_func(mode, dimension1, dimension2)
            data[:, :, imode] = coefficients[imode]*modal_func
        x = np.unique(X)
        y = np.unique(Y)
        return image(data, x, y, float(n))

    @classmethod
    def from_file(image, fpath):
        return io.load_image_file()

    def to_file(self, fpath):
        io.save_image_to_file(self, fpath)

    def transform_data(self, scaling, constant):
        self.data = self.data*scaling + constant

    def add_noise(self, noise_level):
        noise = np.random.normal(0, noise_level, self.data.size).reshape(self.data.shape)
        self.data += noise

    def average_over_dimension(self, dimension='y', use_masked=False):
        if dimension == 'x':
            if use_masked:
                new_data = self.masked_data.mean(axis=0, keepdims=True)
            else:
                new_data = self.data.mean(axis=0, keepdims=True)
            new_x = np.array([0.])
            new_y = self.y
        elif dimension == 'y':
            if use_masked:
                new_data = self.masked_data.mean(axis=1, keepdims=True)
            else:
                new_data = self.data.mean(axis=1, keepdims=True)
            new_x = self.x
            new_y = np.array([0.])
        else:
            raise ValueError("dimension must by either x or y", +
                             " ,value was: {}".format(dimension))
        return LineImage(new_data, new_x, self.z)


    def pixel_comparison(self, other):
        try:
            assert np.all(self.masked_data.shape==other.masked_data.shape)
        except AssertionError as excp:
            print(self.masked_data.shape, other.masked_data.shape)
            raise(excp)

        try:
            assert np.all(self.masked_data.mask == other.masked_data.mask)
        except AssertionError as excp:
            assertion_array = self.masked_data.mask == other.masked_data.mask
            print(np.where(assertion_array==False))
            raise(excp)

        rms_dif = np.sqrt(np.mean((self.masked_data-other.masked_data)**2))
        max_dif = np.max(np.abs(self.masked_data-other.masked_data))
        ls_dif = np.sum((self.masked_data-other.masked_data)**2)
        cubic_dif = np.sum((np.abs(self.masked_data-other.masked_data))**3)
        ls_dif_norm = 0.
        for i_ns in range(self.n_layers):
            ls_dif_norm += np.sum((self.masked_data[:, :, i_ns]-other.masked_data[:, :, i_ns])**2) / np.mean(self.masked_data[:,:, i_ns])**2

        return max_dif, rms_dif, ls_dif, cubic_dif, ls_dif_norm


    def get_closest_index(self, z_pos):
        z_vals = self.z
        pos_diff = np.abs(z_vals-z_pos)
        index = np.where(np.isclose(pos_diff, np.min(pos_diff)))
        #print(index)
        #print(z_vals[index])
        return index




    def central_val(self):
        x_length = self.x.size
        y_length = self.y.size
        x_mid = int((x_length-1)/2)
        y_mid = int((y_length-1)/2)
        z_mid = int((z_length-1)/2)
        return self.masked_data[x_mid, y_mid, z_mid]

    def flux_per_layer(self, xy_scale=1.0):
        X, Y = self.get_cart_dimensions()[:2]
        flux = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            if self.x.size > 1 and self.y.size > 1:
                if self.mask_shape == 'circular':
                    xy = xy_scale*np.vstack([X[:,:,layer].flatten(), Y[:,:,layer].flatten()]).T
                    data_layer = self.masked_data[:,:,layer].flatten()
                    int2D = Integrator2D(xy, data_layer)
                    int_val = int2D.integrate_circle(self.mask_constraint*xy_scale)
                elif self.mask_shape == 'rectangular' or self.mask_shape==None:
                    data_layer = self.masked_data[:,:,layer]
                    int_val = np.trapz(np.trapz(data_layer,
                                                xy_scale*self.x, axis=0),
                                       xy_scale*self.y, axis=0)
            elif self.x.size > 1 and self.y.size == 1:
                data_layer = self.masked_data[:,0,layer]
                int_val = np.trapz(data_layer, xy_scale*self.x)
            elif self.x.size == 1 and self.y.size > 1:
                data_layer = self.masked_data[0,:,layer]
                int_val = np.trapz(data_layer, xy_scale*self.y)
            else:
                raise ValueError("cannot get flux for lateral dimensions of "+
                                 "x,y : {},{}".format(self.x.size, self.y.size))
            flux[layer] = int_val
        return flux

    def normalise_flux(self, xy_scale=1.0):
        flux = self.flux_per_layer(xy_scale=xy_scale)
        for layer in range(self.n_layers):
            self.data[:,:,layer] /= flux[layer]

    def normalise_range(self):
        for layer in range(self.n_layers):
            #xy = np.vstack([self.X[:,:,layer].flatten(), self.Y[:,:,layer].flatten()]).T
            data_layer = self.data[:,:,layer]
            masked_data_layer = self.masked_data[:,:,layer]
            max_val = np.max(masked_data_layer)
            min_val = np.min(masked_data_layer)
            #int2D = Integrator2D(xy, data_layer)
            #int_val = int2D.integrate_circle(self.mask_radius)
            if np.isclose(np.abs(max_val-min_val), 0.):
                continue
            new_data_layer = (data_layer-min_val)/(max_val-min_val)
            self.data[:,:,layer] = new_data_layer

    def normalise_highest(self):
        max_val = np.max(self.masked_data)
        self.data /= max_val

    def slice_z(self, z_vals):
        current_z = self.z
        data_stack = np.zeros((self.data.shape[0], self.data.shape[1],
                               z_vals.size))

        for iz1, z1 in enumerate(z_vals):
            is_sliced = False
            for iz2, z2 in enumerate(current_z):
                if np.isclose(z1,z2,rtol=1e-3, atol=3e-9):
                    is_sliced = True
                    data_stack[:, :, iz1] = self.data[:, :, iz2]
            #print("{} is sliced: {}".format(z1, is_sliced))

        #x = self.X[:,0,0]
        #y = self.Y[0,:,0]
        x = self.x
        y = self.y
        #X, Y, Z = np.meshgrid(x, y, z_vals, indexing='ij')
        sliced_im_stack = ImageStack(data_stack, x, y, z_vals)
        if self.mask_shape is not None:
            sliced_im_stack.apply_mask(self.mask_shape, self.mask_region,
                                       self.mask_constraint)
        return sliced_im_stack

    def projection(self, image):
        n_unmasked = ma.count(image)
        coefficients = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            basis_func = self.masked_data[:,:,layer]
            coef = ma.sum(image*basis_func)/n_unmasked
            coefficients[layer] = coef
        return coefficients

    def sum(self):
        return np.sum(self.masked_data,axis=2)

    def optimise_n_coefficients(self, max_n_modes_limit,
                                basis='zernike_fringe_constrained'):
        """
        Find the optimal number of basis coefficients for each layer

        For each layer of the ImageStack, we minimize the information
        criterion for a given set of basis functions to find how many functions
        are required to give a good fit to the data without overfitting.

        Parameters
        ----------
        max_n_modes_limit: int
            upper limit on number of basis coefficients

        """
        max_n_modes_required = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            minimize_func = lambda x : self.information_criterion(x,
                                                                  basis,
                                                                  layer)
            res = scipy.optimize.minimize_scalar(minimize_func,
                                           bounds=(20, max_n_modes_limit),
                                           method='bounded',
                                           options={'maxiter':20,
                                                    'xatol':0.5,
                                                    'disp':True})
            max_n_modes_required[layer] = int(res['x'])
            print(layer)
        return max_n_modes_required

    def information_criterion(self, max_n_modes, basis_name, layer,
                              criterion='bic'):
        max_n_modes = int(max_n_modes)
        outputs = self.fit_parameters(max_n_modes=max_n_modes,
                                      basis=basis_name,
                                      layers=[layer])
        residual = outputs[5][0] + 1e-10
        n = ma.count(self.masked_data[:,:,layer])
        if criterion == 'bic':
            criterion = np.log(n)*max_n_modes +  n*np.log(residual)
        elif criterion == 'aic':
            criterion = 2*max_n_modes +  n*np.log(residual)
        return criterion

    def std_per_layer(self):

        if self.y.size == 1:
            std = np.std(self.masked_data,axis=0)[0,:]
        elif self.x.size == 1:
            std = np.std(self.masked_data,axis=1)[0,:]
        else:
            std = np.std(self.masked_data, axis=(0,1))
        return std

    def max_val_per_layer(self):

        if self.y.size == 1:
            max_val = np.max(self.masked_data,axis=0)[0,:]
        elif self.x.size == 1:
            max_val = np.max(self.masked_data,axis=1)[0,:]
        else:
            max_val = np.max(self.masked_data, axis=(0,1))
        return max_val


    def determine_focal_plane(self, axes=None, center_on_focal=False,
                              shrink_window=1.0, center_on_nearest=False,
                              fit_parameter='intensity', fit_method='iterative_polynomial'):
        """
        Determine the focal plane via polynomial fitting to the data

        Parameters
        ----------
        axes: matplotlib.axes._subplots.AxesSubplot
            axes to plot the fitting process in
        center_on_focal: bool
            change the z class attribute to have focal position at 0.
        """

        current_guess = 'center'
        focal_pos = 0.
        if axes is not None:
            plt.sca(axes)
            if fit_parameter == 'intensity':
                fit_parameter_per_plane = np.max(np.max(self.masked_data,axis=0), axis=0)
            elif fit_parameter == 'std_dev':
                fit_parameter_per_plane = self.std_per_layer()
            plt.plot(self.z, fit_parameter_per_plane, marker='.', label='Exp. Data')

        focal_plane_found = False
        if fit_method == 'iterative_polynomial':
            current_guess = {'focal_plane_index':current_guess,
                             'poly_deg':3}
        elif fit_method == 'gaussian':
            current_guess = {'focal_plane_index':current_guess,
                             'std_dev':None}


        while not focal_plane_found:
            guess_area_width = shrink_window*self.n_layers
            """
            if isinstance(current_guess['focal_plane_index'], str):
                guess_area_width = shrink_window*self.n_layers
            else:
                guess_area_width = shrink_window*np.min([self.n_layers,
                                                         2*current_guess['focal_plane_index'],
                                                         2*(self.n_layers-current_guess['focal_plane_index'])])
            """
            current_guess['area_width'] = guess_area_width
            if fit_method == 'iterative_polynomial':
                fit_data = self.fit_polynomial_to_data(current_guess,
                                                       fit_parameter=fit_parameter)
            elif fit_method == 'gaussian':
                fit_data = self.fit_gaussian_to_data(current_guess,
                                                     fit_parameter=fit_parameter)
                focal_plane_found = True
            new_focal_pos = fit_data[0]
            focal_plane_nearest_index = fit_data[1]
            z_pos_for_polyfit = fit_data[2]
            max_val_fitted = fit_data[3]
            #print(current_poly_deg, current_guess, guess_area_width, new_focal_pos, focal_plane_nearest_index )
            if axes is not None:
                if fit_method == 'iterative_polynomial':
                    plt.plot(z_pos_for_polyfit, max_val_fitted,
                            label='Poly. Deg. {}'.format(current_guess['poly_deg']))
                elif fit_method == 'gaussian':
                    plt.plot(z_pos_for_polyfit, max_val_fitted,
                            label='Gauss Fit')
            #print("isclose: {}, {}, {}".format(new_focal_pos, focal_pos, np.isclose(new_focal_pos, focal_pos, rtol=1e-3, atol=1e-12)))
            if fit_method == 'iterative_polynomial':
                if np.isclose(new_focal_pos, focal_pos, atol=1e-9):
                    focal_plane_found = True
                    focal_pos = new_focal_pos
                else:
                    focal_pos = new_focal_pos
                    current_guess['poly_deg'] += 1
                    current_guess['focal_plane_index'] = focal_plane_nearest_index
            elif fit_method == 'gaussian':
                focal_pos = new_focal_pos
        focal_pos = np.round(focal_pos, 11)
        self.focal_plane_nearest_index = focal_plane_nearest_index
        if center_on_focal:
            self.z = self.z-focal_pos
            self.focal_plane = 0.
        elif center_on_nearest:
            nearest_z = self.z[self.focal_plane_nearest_index]
            self.z = self.z-nearest_z
            self.focal_plane = focal_pos-nearest_z
        else:
            self.focal_plane = focal_pos
        self.focal_plane_nearest_value = self.z[self.focal_plane_nearest_index]
        return focal_pos

    def _pre_process_guess(self, guess):
        """
        convert guess focal plane index 'center' to the central layer of stack

        Parameters
        ----------
        guess: dict
            the guess with field 'focal_plane_index'

        """
        if isinstance(guess['focal_plane_index'],str):
            if 'center' == guess['focal_plane_index']:
                guess['focal_plane_index'] = int((self.n_layers-1)/2.)
            else:
                raise ValueError("unknown focal plane starting guess with value:" +
                                 guess['focal_plane_index'])
        elif isinstance(guess['focal_plane_index'], (int, np.integer)):
            guess['focal_plane_index'] = guess['focal_plane_index']
        else:
            raise TypeError("focal plane starting guess must be integer or string")

    def _get_fit_param(self, fit_parameter):
        """
        Return a the fitting parameter per z plane

        Parameters
        ----------
        fit_parameter: str
            the parameter used for fitting

        """
        if fit_parameter == 'intensity':
            fit_param_per_plane = np.max(np.max(self.masked_data,axis=0), axis=0)
        elif fit_parameter == 'std_dev':
            fit_param_per_plane = self.std_per_layer()
        return fit_param_per_plane


    def fit_polynomial_to_data(self, guess, fit_parameter='intensity'):
        """
        Fit polynomial to the maximum value per plane.

        We seek the plane with the highest intensity value. At the same time,
        noisy data is assumed. We fit the max values per plane using a polynomial
        and take the maximum of this to avoid noise.

        Parameters
        ----------
        guess: str or int
            guess for the index nearest the focal plane, if "center" is given
            as input
        guess_area_width: int
            width in values of index to look around the focal plane guess
        polydef: int
            degree of polynomial fit to use
        """
        self._pre_process_guess(guess)
        guess_area_start = np.max([int(guess['focal_plane_index']-guess['area_width']/2),0])
        guess_area_end = np.min([int(guess['focal_plane_index']+guess['area_width']/2),self.z.size-1])
        fit_param_per_plane = self._get_fit_param(fit_parameter)
        fit_param_per_plane_sliced = fit_param_per_plane[guess_area_start:guess_area_end]
        z_pos_sliced = self.z[guess_area_start:guess_area_end]
        z_pos_for_polyfit = np.linspace( np.min(z_pos_sliced), np.max(z_pos_sliced), 10000)
        coeffs = np.polyfit(z_pos_sliced, fit_param_per_plane_sliced, guess['poly_deg'])
        fit_param_fitted = np.zeros(z_pos_for_polyfit.shape)
        for ii, coeff in enumerate(coeffs[::-1]):
            fit_param_fitted += coeff*np.power(z_pos_for_polyfit, ii)

        fitted_index_for_max = np.where(fit_param_fitted==np.max(fit_param_fitted))[0]
        focal_pos = z_pos_for_polyfit[fitted_index_for_max][0] # the z value for the maximum of the polyfit

        diff_to_focal_plane = np.abs(z_pos_sliced-focal_pos)
        focal_plane_index_nearest = guess_area_start + np.where(diff_to_focal_plane==np.min(diff_to_focal_plane))[0][0] # the index that is closest to the maximum

        return (focal_pos, focal_plane_index_nearest, z_pos_for_polyfit,
                fit_param_fitted)


    def fit_gaussian_to_data(self, guess, fit_parameter='intensity'):
        """
        Fit polynomial to the maximum value per plane.

        We seek the plane with the highest intensity value. At the same time,
        noisy data is assumed. We fit the max values per plane using a polynomial
        and take the maximum of this to avoid noise.

        Parameters
        ----------
        guess: str or int
            guess for the index nearest the focal plane, if "center" is given
            as input
        guess_area_width: int
            width in values of index to look around the focal plane guess
        polydef: int
            degree of polynomial fit to use
        """
        self._pre_process_guess(guess)
        guess_area_start = np.max([int(guess['focal_plane_index']-guess['area_width']/2),0])
        guess_area_end = np.min([int(guess['focal_plane_index']+guess['area_width']/2),self.z.size-1])
        fit_param_per_plane = self._get_fit_param(fit_parameter)
        fit_param_per_plane_sliced = fit_param_per_plane[guess_area_start:guess_area_end]
        z_pos_sliced = self.z[guess_area_start:guess_area_end]
        guess_std_dev = (z_pos_sliced[-1]-z_pos_sliced[0])*0.25
        start_vals = (np.max(fit_param_per_plane_sliced),
                      self.z[guess['focal_plane_index']+1], guess_std_dev)
        minimize_func = lambda x : utils.gaussian(x, z_pos_sliced)-fit_param_per_plane_sliced
        res = scipy.optimize.leastsq(minimize_func,
                                     x0=start_vals,
                                     full_output=False)
        z_pos_for_fit = np.linspace( np.min(z_pos_sliced), np.max(z_pos_sliced), 10000)
        fit_param_fitted = utils.gaussian(res[0], z_pos_for_fit)

        fitted_index_for_max = np.where(fit_param_fitted==np.max(fit_param_fitted))[0]
        focal_pos = z_pos_for_fit[fitted_index_for_max][0] # the z value for the maximum of the fit

        diff_to_focal_plane = np.abs(z_pos_sliced-focal_pos)
        focal_plane_index_nearest = guess_area_start + np.where(diff_to_focal_plane==np.min(diff_to_focal_plane))[0][0] # the index that is closest to the maximum
        return (focal_pos, focal_plane_index_nearest, z_pos_for_fit,
                fit_param_fitted)

    def max_afocal_range(self):
        left_range = np.abs(self.z[0]-self.focal_plane)
        right_range = np.abs(self.z[-1]-self.focal_plane)
        return np.min([left_range, right_range])


    def fit_parameters(self, basis='zernike', max_n_modes=10,
                       n_modes_per_layer=None, layers='all'):
        if basis == 'zernike_fringe' or basis == 'zernike':
            basis_func = basis_functions.zernike_fringe
            polar = True
            modes_start = 1
        elif basis == 'zernike_osa':
            basis_func = basis_functions.zernike_osa
            polar = True
            modes_start = 1
        elif basis == 'zernike_fringe_constrained':
            basis_func = basis_functions.zernike_fringe_constrained
            polar = True
            modes_start = 1
        elif basis == 'legendre':
            basis_func = basis_functions.legendre
            polar = False
            modes_start = 0
        else:
            raise ValueError("basis func: {} unknown".format(basis))

        if n_modes_per_layer is not None:
            max_n_modes = np.max(n_modes_per_layer)

        if layers == 'all':
            layers = range(self.n_layers)
        n_layers = len(layers)
        X, Y = self.get_cart_dimensions()[:2]
        if polar:
            X = X[:, :, 0]/self.mask_constraint
            Y = Y[:, :, 0]/self.mask_constraint
        else:
            X = X[:, :, 0]/self.mask_constraint[0]
            Y = Y[:, :, 0]/self.mask_constraint[1]
        image0 = self.masked_data[:,:,0]
        bd = BasisDecomposition(image0, basis_func, max_n_modes, modes_start,
                                X, Y, self.mask_region, polar)

        all_projected_coefficients = np.zeros((n_layers, max_n_modes))
        all_fitted_coefficients = np.zeros((n_layers, max_n_modes))
        all_std_devs = np.zeros((n_layers, max_n_modes))
        squared_residuals = np.zeros((n_layers))

        projected_data = ImageStack(np.zeros_like(self.data), self.x, self.y, self.z)
        fitted_data = ImageStack(np.zeros_like(self.data), self.x, self.y, self.z)
        #projected_slices = []
        #fitted_slices = []
        res = None
        for ilayer, layer in enumerate(layers):
            image = self.masked_data[:,:,layer]
            if n_modes_per_layer is not None:
                current_max_n_modes = n_modes_per_layer[ilayer]
            else:
                current_max_n_modes = max_n_modes
            bd.set_image(image)
            bd.set_current_max_n_modes(current_max_n_modes)
            projected_coefficients = bd.project_coefficients()[:current_max_n_modes]
            all_projected_coefficients[ilayer, :current_max_n_modes] = projected_coefficients
            projected_data.data[:,:, layer] = bd.image_from_coefficients(projected_coefficients)
            res = bd.fit_coefficients(projected_coefficients)
            all_fitted_coefficients[ilayer,:current_max_n_modes] = res[0]
            #res[1][np.isnan(res[1])] = 0.
            if res[1] is not None:
                diag_vals = np.squeeze(np.array([np.diag(res[1])]))
                diag_vals[diag_vals<0.] = 0.
                all_std_devs[ilayer, :current_max_n_modes] = np.sqrt(diag_vals+1e-10)
            fitted_data.data[:,:,layer] = bd.image_from_coefficients(res[0])
            squared_residuals[ilayer] = (res[2]['fvec']**2).sum()

        return (projected_data, all_projected_coefficients, res, projected_data,
                fitted_data, squared_residuals, all_fitted_coefficients, all_std_devs)

        #return coefficients




class ImageStack():

    """
    Analysis of layers of pixel images in different focal planes.

    Attributes
    ---------
    data: (N,M,P) np.ndarray
        The pixel data
    masked_array: (N,M,P) np.ma.maskedarray
        The masked pixel data
    x: (N,) np.ndarray
        The positions of the pixels in the x direction
    y: (M,) np.ndarray
        The positions of the pixels in the y direction
    z: (P,) np.ndarray
        The positions of the layers in the z direction
    n_layers: int
        The number of layers of images
    mask_shape: str
        The shape of the mask for data parameter fitting
    mask_region: str
        The region of the mask for data parameter fitting
    mask_constraint: float or (2,) tuple
        constraint for the mask for data parameter fitting
    focal_plane_nearest_index: int
        the index of the z position closest to the focal plane position
    focal_plane_nearest_value: float
        the z position closest to the focal plane position
    focal_plane: float
        the z position of the focal plane
    label: str
        label used for plotting
    """

    def __init__(self, data, x, y, z):
        """
        Parameters
        ----------
        data: (N,M,P) np.ndarray
            The pixel data
        x: (N,) np.ndarray
            The positions of the pixels in the x direction
        y: (M,) np.ndarray
            The positions of the pixels in the y direction
        z: (P,) np.ndarray
            The positions of the layers in the z direction
        """
        self.data = data
        self.masked_data = ma.array(data, copy=False)
        self.x = x
        self.y = y
        self.z = z
        self.n_layers = self.z.size

        self.mask_shape = None
        self.mask_region = None
        self.mask_constraint = 1.0

        self.focal_plane_nearest_index = None
        self.focal_plane_nearest_value = None
        self.focal_plane = 0.

        self.label=""

    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')

    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        X, Y, Z = self.get_cart_dimensions()
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        return R, PHI, Z

    @classmethod
    def from_matlab_data(image_stack, path, pixel_size='from_file',
                        slice_distance='from_file'):
        """
        Constructor for ImageStack from a Matlab .mat file

        Parameters
        ----------
        path: str
            path to a Matlab .mat file
        pixel_size: str or float
            distance between pixels (x,y direction) in m
        slice_distance: str or float
            distance between layers of the stack in m
        """
        matlab_data = scipy.io.loadmat(path)
        experimental = matlab_data['image_data']

        if pixel_size == 'from_file':
            pixel_size = matlab_data['pixel_size']

        x_length = experimental.shape[0]
        x_half_length = (x_length-1)/2 #assumes 0 centered data
        max_x = pixel_size[0][0]*x_half_length
        x = np.linspace(-max_x, max_x, x_length)

        y_length = experimental.shape[1]
        y_half_length = (y_length-1)/2 #assumes 0 centered data
        max_y = pixel_size[0][0]*y_half_length
        y = np.linspace(-max_y, max_y, y_length)

        if slice_distance == 'from_file':
            z = np.squeeze(matlab_data['z'])
        else:
            z_length = experimental.shape[2]
            z_half_length = (z_length-1)/2 #assumes 0 centered data
            max_z = slice_distance*z_half_length
            z = np.linspace(-max_z, max_z, z_length)

        #X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return image_stack(experimental, x, y, z)

    @classmethod
    def from_cartesian_fields(image_stack, cart_fields, z, xy_scale=1.0):
        x_shape = cart_fields[0]['X'].shape
        xy_length = x_shape[0]
        xy_mid = int((xy_length-1)/2)
        n_layers = len(cart_fields)
        optical_images = np.zeros((xy_length, xy_length, n_layers))
        for z_index in range(n_layers):
            cart_field = cart_fields[z_index]
            if z_index == 0:
                x = np.unique(cart_field['X'])*xy_scale
                y = np.unique(cart_field['Y'])*xy_scale
                #Z = cart_field['Z']
            n_pol = len(cart_field['field'])
            intensity = np.zeros((x_shape))
            for i_pol in range(n_pol):
                if cart_field['field'][i_pol].shape[-1] == 3:
                    #print("vector field detected")
                    intensity += np.linalg.norm(cart_field['field'][i_pol], axis=2)**2
                else:
                    #print(intensity.shape)
                    #print("scalar field detected")
                    update = 4.*np.squeeze(np.real(cart_field['field'][i_pol]))/scipy.constants.epsilon_0
                    #print(update.shape)
                    intensity += update
            intensity /= n_pol
            optical_images[:, :, z_index] = intensity

        #X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return image_stack(optical_images, x, y, z)

    @classmethod
    def from_basis(image_stack, basis_func, n_modes, mode_start, X, Y, polar=False,
                   coefficients=None):
        data = np.zeros((X.shape[0], Y.shape[1], n_modes))
        if coefficients is None:
            coefficients = np.ones(n_modes)
        mode_range = range(mode_start, n_modes+mode_start)
        if polar:
            R = np.sqrt(X**2 + Y**2)
            PHI = np.arctan2(Y,X)
            dimension1 = R
            dimension2 = PHI
        else:
            dimension1 = X
            dimension2 = Y
        len_x = X.shape[0]
        len_y = Y.shape[1]
        for imode, mode in enumerate(mode_range):
            modal_func = basis_func(mode, dimension1, dimension2)
            data[:, :, imode] = coefficients[imode]*modal_func
        x = np.unique(X)
        y = np.unique(Y)
        z = np.array(list(mode_range))

        #X, Y, Z = np.meshgrid(x, y, z)

        return image_stack(data, x, y, z)

    @classmethod
    def from_file(image_stack, fpath):
        with open(fpath, 'rb') as fp:
            unpickler = pickle.Unpickler(fp)
            im_stack = unpickler.load()
        return im_stack

    @classmethod
    def concatenate(image_stack, im_stack_list):
        im_stack0 = im_stack_list[0]
        n_layers = 0
        for stack in im_stack_list:
            n_layers += im_stack0.z.size
        data_shape = (im_stack0.data.shape[0], im_stack0.data.shape[1], n_layers)
        new_data = np.zeros(data_shape)
        ii = 0
        z = []
        for i_stack, stack in enumerate(im_stack_list):
            current_n_layers = im_stack0.z.size
            z.append(np.unique(stack.z))
            new_data[:,:,ii:ii+current_n_layers] = stack.data
            ii += current_n_layers
        z = np.concatenate(z)
        x = np.unique(im_stack0.x)
        y = np.unique(im_stack0.y)
        return ImageStack(new_data, x, y, z)


    def to_file(self, fpath):
        with open(fpath, 'wb') as fp:
            pickler = pickle.Pickler(fp)
            pickler.dump(self)

    def transform_data(self, scaling, constant):
        self.data = self.data*scaling + constant

    def add_noise_floor(self, floor):
        floor_array = floor*np.ones(self.data.shape)
        self.data =  np.maximum(self.data, floor_array)

    def add_noise(self, noise_level):
        noise = np.random.normal(0, noise_level, self.data.size).reshape(self.data.shape)
        self.data += noise

    def apply_mask(self, shape, region, constraint):
        if shape == 'circular':
            mask = self._get_circular_mask(region, constraint)
        elif shape == 'rectangular':
            mask = self._get_rectangular_mask(region, constraint)
        else:
            raise ValueError("unknown mask type :[{}]".format(shape))
        self.mask_shape = shape
        self.mask_region = region
        self.mask_constraint = constraint
        self.masked_data.mask = mask

    def _get_circular_mask(self, region, radius):
        R = self.get_cyl_dimensions()[0]
        if region == 'edge':
            mask = R<radius
        elif region =='center':
            mask = R>radius
        elif region =='none':
            mask = np.full(R.shape, False, dtype=bool)
        else:
            raise ValueError("mask region name {} invalid".format(region))
        return mask

    def _get_rectangular_mask(self, region, xy_width):
        X, Y = self.get_cart_dimensions()[:2]
        if region == 'edge':
            mask = np.abs(X) < xy_width[0]
            mask += np.abs(Y) < xy_width[1]
        elif region =='center':
            mask = np.abs(X) > xy_width[0]
            mask += np.abs(Y) > xy_width[1]
        elif region =='none':
            mask = np.full(X.shape, False, dtype=bool)
        else:
            raise ValueError("mask region name {} invalid".format(region))
        return mask

    def plot_image(self, ax, z_index=None, z_pos=None, vmin=None, vmax=None,
                   xy_scale=1.0, colorbar=True):
        if z_index is None and z_pos is None:
            z_index = int((self.n_layers-1)/2)
        elif z_pos is not None:
            z_index = self.get_closest_index(z_pos)
        X, Y = self.get_cart_dimensions()[:2]
        image_slice = np.squeeze(self.masked_data[:, :, z_index])
        X = np.squeeze(X[:, :, z_index])*xy_scale
        Y = np.squeeze(Y[:, :, z_index])*xy_scale
        plt.sca(ax)
        plt.pcolormesh(X, Y, image_slice, shading='gouraud',
                       vmin=vmin, vmax=vmax)
        if colorbar:
            plt.colorbar()
        ax.axis('equal')

    def plot_line(self, ax, z_index=None, z_pos=None, x_scale=1.0):
        if z_index is None and z_pos is None:
            z_index = int((self.n_layers-1)/2)
        elif z_pos is not None:
            z_index = self.get_closest_index(z_pos)
        X = self.get_cart_dimensions()[0]
        image_slice = self.masked_data[:, :, z_index]
        if self.y.size > 1:
            line_image = np.squeeze(image_slice.mean(axis=1))
        else:
            line_image = image_slice
        X = np.squeeze(X[:, 0, z_index])*x_scale
        plt.sca(ax)
        plt.plot(X, line_image, label=self.label)

    def average_over_dimension(self, dimension='y'):
        if dimension == 'x':
            new_data = self.data.mean(axis=0, keepdims=True)
            new_x = np.array([0.])
            new_y = self.y
        elif dimension == 'y':
            #print(type(self.data))
            new_data = self.data.mean(axis=1, keepdims=True)
            new_x = self.x
            new_y = np.array([0.])
        return ImageStack(new_data, new_x, new_y, self.z)


    def pixel_comparison(self, other):
        try:
            assert np.all(self.masked_data.shape==other.masked_data.shape)
        except AssertionError as excp:
            print(self.masked_data.shape, other.masked_data.shape)
            raise(excp)

        try:
            assert np.all(self.masked_data.mask == other.masked_data.mask)
        except AssertionError as excp:
            assertion_array = self.masked_data.mask == other.masked_data.mask
            print(np.where(assertion_array==False))
            raise(excp)

        rms_dif = np.sqrt(np.mean((self.masked_data-other.masked_data)**2))
        max_dif = np.max(np.abs(self.masked_data-other.masked_data))
        ls_dif = np.sum((self.masked_data-other.masked_data)**2)
        cubic_dif = np.sum((np.abs(self.masked_data-other.masked_data))**3)
        ls_dif_norm = 0.
        for i_ns in range(self.n_layers):
            ls_dif_norm += np.sum((self.masked_data[:, :, i_ns]-other.masked_data[:, :, i_ns])**2) / np.mean(self.masked_data[:,:, i_ns])**2

        return max_dif, rms_dif, ls_dif, cubic_dif, ls_dif_norm


    def get_closest_index(self, z_pos):
        z_vals = self.z
        pos_diff = np.abs(z_vals-z_pos)
        index = np.where(np.isclose(pos_diff, np.min(pos_diff)))
        #print(index)
        #print(z_vals[index])
        return index

    def plot_image_difference(self, other, path, fname, ext='png',
                              to_file=True, other2=None, layer_step=5):
        if self.n_layers == 1:
            ext = 'png'

        if to_file:
            if ext == 'pdf':
                pdf_filename = fname + ".{}".format(ext)
                pdf = PdfPages(os.path.join(path, pdf_filename))
                dpi = 50
            elif ext =='png':
                dpi = 300

        max_distance_left = self.focal_plane_nearest_index
        max_distance_right = self.n_layers-self.focal_plane_nearest_index
        print(max_distance_left, max_distance_right)
        max_distance = np.min([max_distance_left, max_distance_right])
        step_size = layer_step
        layers_right = np.arange(0, max_distance+step_size, step_size)
        layers_left = -1*layers_right[:0:-1]
        layers = np.concatenate([layers_left, layers_right]) + self.focal_plane_nearest_index
        print(layers)
        X, Y = self.get_cart_dimensions()[:2]
        X1 = X[:,:,0]
        Y1 = Y[:,:,0]

        X, Y = other.get_cart_dimensions()[:2]
        X2 = X[:,:,0]
        Y2 = Y[:,:,0]
        for layer in layers:
            z_pos = self.z[layer]
            print(layer)
            print(z_pos)
            self_slice = self.masked_data[:, :, layer]
            other_layer = other.get_closest_index(z_pos)
            other_slice = np.squeeze(other.masked_data[:, :, other_layer])
            print(self_slice.shape)
            print(other_slice.shape)

            #return None
            #X = self.X[:, :, layer]
            #Y = self.Y[:, :, layer]
            #vmin = np.min([np.min(self_slice), np.min(other_slice)])
            #vmax = np.max([np.max(self_slice), np.max(other_slice)])

            un_masked = other.masked_data.compressed()
            nans = un_masked[np.isnan(un_masked)]

            #print(nans.size)
            #print([np.min(self.data.compressed()), np.min(other.data.compressed())])
            #print([np.max(self.data.compressed()), np.max(other.data.compressed())])
            #print([np.min(self_slice), np.min(other_slice)])
            #print([np.max(self_slice), np.max(other_slice)])
            vmin = np.min([np.min(self_slice.compressed()), np.min(other_slice.compressed())])
            vmax = np.max([np.max(self_slice.compressed()), np.max(other_slice.compressed())])
            print(vmin, vmax)
            if other2 is not None:
                other2_slice = other2.masked_data[:, :, layer]
                vmin = np.min([vmin, np.min(other2_slice)])
                vmax = np.max([vmax, np.max(other2_slice)])
                vmin = None
                vmax = None
                fig = triple_plot(X, Y, self_slice, other_slice, other2_slice,
                                vmax=vmax, vmin=vmin)
            else:
                fig = combined_plot((X1, X2), (Y1, Y2), self_slice, other_slice,
                                vmax=vmax, vmin=vmin,
                                label1=self.label, label2=other.label)

            suptitle = "i$_{z}$:"+"{}, z: {:.2f} $\mu$m".format(layer, z_pos*1e6)
            plt.suptitle(suptitle, fontsize=20.)
            if to_file:
                if ext == 'png':
                    filename = fname + "_{}".format(layer) + ".{}".format(ext)
                    plt.savefig(os.path.join(path, filename), dpi=dpi, bbox_inches='tight')
                elif ext == 'pdf':
                    pdf.savefig(fig, dpi=dpi)
                plt.close(fig)

        if ext == 'pdf' and to_file:
            pdf.close()
        return fig



    def central_val(self):
        x_length = self.x.size
        y_length = self.y.size
        z_length = self.z.size
        x_mid = int((x_length-1)/2)
        y_mid = int((y_length-1)/2)
        z_mid = int((z_length-1)/2)
        return self.masked_data[x_mid, y_mid, z_mid]

    def flux_per_layer(self, xy_scale=1.0):
        X, Y = self.get_cart_dimensions()[:2]
        flux = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            if self.x.size > 1 and self.y.size > 1:
                if self.mask_shape == 'circular':
                    xy = xy_scale*np.vstack([X[:,:,layer].flatten(), Y[:,:,layer].flatten()]).T
                    data_layer = self.masked_data[:,:,layer].flatten()
                    int2D = Integrator2D(xy, data_layer)
                    int_val = int2D.integrate_circle(self.mask_constraint*xy_scale)
                elif self.mask_shape == 'rectangular' or self.mask_shape==None:
                    data_layer = self.masked_data[:,:,layer]
                    int_val = np.trapz(np.trapz(data_layer,
                                                xy_scale*self.x, axis=0),
                                       xy_scale*self.y, axis=0)
            elif self.x.size > 1 and self.y.size == 1:
                data_layer = self.masked_data[:,0,layer]
                int_val = np.trapz(data_layer, xy_scale*self.x)
            elif self.x.size == 1 and self.y.size > 1:
                data_layer = self.masked_data[0,:,layer]
                int_val = np.trapz(data_layer, xy_scale*self.y)
            else:
                raise ValueError("cannot get flux for lateral dimensions of "+
                                 "x,y : {},{}".format(self.x.size, self.y.size))
            flux[layer] = int_val
        return flux

    def normalise_flux(self, xy_scale=1.0):
        flux = self.flux_per_layer(xy_scale=xy_scale)
        for layer in range(self.n_layers):
            self.data[:,:,layer] /= flux[layer]

    def normalise_range(self):
        for layer in range(self.n_layers):
            #xy = np.vstack([self.X[:,:,layer].flatten(), self.Y[:,:,layer].flatten()]).T
            data_layer = self.data[:,:,layer]
            masked_data_layer = self.masked_data[:,:,layer]
            max_val = np.max(masked_data_layer)
            min_val = np.min(masked_data_layer)
            #int2D = Integrator2D(xy, data_layer)
            #int_val = int2D.integrate_circle(self.mask_radius)
            if np.isclose(np.abs(max_val-min_val), 0.):
                continue
            new_data_layer = (data_layer-min_val)/(max_val-min_val)
            self.data[:,:,layer] = new_data_layer

    def normalise_highest(self):
        max_val = np.max(self.masked_data)
        self.data /= max_val

    def slice_z(self, z_vals):
        current_z = self.z
        data_stack = np.zeros((self.data.shape[0], self.data.shape[1],
                               z_vals.size))

        for iz1, z1 in enumerate(z_vals):
            is_sliced = False
            for iz2, z2 in enumerate(current_z):
                if np.isclose(z1,z2,rtol=1e-3, atol=3e-9):
                    is_sliced = True
                    data_stack[:, :, iz1] = self.data[:, :, iz2]
            #print("{} is sliced: {}".format(z1, is_sliced))

        #x = self.X[:,0,0]
        #y = self.Y[0,:,0]
        x = self.x
        y = self.y
        #X, Y, Z = np.meshgrid(x, y, z_vals, indexing='ij')
        sliced_im_stack = ImageStack(data_stack, x, y, z_vals)
        if self.mask_shape is not None:
            sliced_im_stack.apply_mask(self.mask_shape, self.mask_region,
                                       self.mask_constraint)
        return sliced_im_stack

    def projection(self, image):
        n_unmasked = ma.count(image)
        coefficients = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            basis_func = self.masked_data[:,:,layer]
            coef = ma.sum(image*basis_func)/n_unmasked
            coefficients[layer] = coef
        return coefficients

    def sum(self):
        return np.sum(self.masked_data,axis=2)

    def optimise_n_coefficients(self, max_n_modes_limit,
                                basis='zernike_fringe_constrained'):
        """
        Find the optimal number of basis coefficients for each layer

        For each layer of the ImageStack, we minimize the information
        criterion for a given set of basis functions to find how many functions
        are required to give a good fit to the data without overfitting.

        Parameters
        ----------
        max_n_modes_limit: int
            upper limit on number of basis coefficients

        """
        max_n_modes_required = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            minimize_func = lambda x : self.information_criterion(x,
                                                                  basis,
                                                                  layer)
            res = scipy.optimize.minimize_scalar(minimize_func,
                                           bounds=(20, max_n_modes_limit),
                                           method='bounded',
                                           options={'maxiter':20,
                                                    'xatol':0.5,
                                                    'disp':True})
            max_n_modes_required[layer] = int(res['x'])
            print(layer)
        return max_n_modes_required

    def information_criterion(self, max_n_modes, basis_name, layer,
                              criterion='bic'):
        max_n_modes = int(max_n_modes)
        outputs = self.fit_parameters(max_n_modes=max_n_modes,
                                      basis=basis_name,
                                      layers=[layer])
        residual = outputs[5][0] + 1e-10
        n = ma.count(self.masked_data[:,:,layer])
        if criterion == 'bic':
            criterion = np.log(n)*max_n_modes +  n*np.log(residual)
        elif criterion == 'aic':
            criterion = 2*max_n_modes +  n*np.log(residual)
        return criterion

    def std_per_layer(self):

        if self.y.size == 1:
            std = np.std(self.masked_data,axis=0)[0,:]
        elif self.x.size == 1:
            std = np.std(self.masked_data,axis=1)[0,:]
        else:
            std = np.std(self.masked_data, axis=(0,1))
        return std

    def max_val_per_layer(self):

        if self.y.size == 1:
            max_val = np.max(self.masked_data,axis=0)[0,:]
        elif self.x.size == 1:
            max_val = np.max(self.masked_data,axis=1)[0,:]
        else:
            max_val = np.max(self.masked_data, axis=(0,1))
        return max_val


    def determine_focal_plane(self, axes=None, center_on_focal=False,
                              shrink_window=1.0, center_on_nearest=False,
                              fit_parameter='intensity', fit_method='iterative_polynomial'):
        """
        Determine the focal plane via polynomial fitting to the data

        Parameters
        ----------
        axes: matplotlib.axes._subplots.AxesSubplot
            axes to plot the fitting process in
        center_on_focal: bool
            change the z class attribute to have focal position at 0.
        """

        current_guess = 'center'
        focal_pos = 0.
        if axes is not None:
            plt.sca(axes)
            if fit_parameter == 'intensity':
                fit_parameter_per_plane = np.max(np.max(self.masked_data,axis=0), axis=0)
            elif fit_parameter == 'std_dev':
                fit_parameter_per_plane = self.std_per_layer()
            plt.plot(self.z, fit_parameter_per_plane, marker='.', label='Exp. Data')

        focal_plane_found = False
        if fit_method == 'iterative_polynomial':
            current_guess = {'focal_plane_index':current_guess,
                             'poly_deg':3}
        elif fit_method == 'gaussian':
            current_guess = {'focal_plane_index':current_guess,
                             'std_dev':None}


        while not focal_plane_found:
            guess_area_width = shrink_window*self.n_layers
            """
            if isinstance(current_guess['focal_plane_index'], str):
                guess_area_width = shrink_window*self.n_layers
            else:
                guess_area_width = shrink_window*np.min([self.n_layers,
                                                         2*current_guess['focal_plane_index'],
                                                         2*(self.n_layers-current_guess['focal_plane_index'])])
            """
            current_guess['area_width'] = guess_area_width
            if fit_method == 'iterative_polynomial':
                fit_data = self.fit_polynomial_to_data(current_guess,
                                                       fit_parameter=fit_parameter)
            elif fit_method == 'gaussian':
                fit_data = self.fit_gaussian_to_data(current_guess,
                                                     fit_parameter=fit_parameter)
                focal_plane_found = True
            new_focal_pos = fit_data[0]
            focal_plane_nearest_index = fit_data[1]
            z_pos_for_polyfit = fit_data[2]
            max_val_fitted = fit_data[3]
            #print(current_poly_deg, current_guess, guess_area_width, new_focal_pos, focal_plane_nearest_index )
            if axes is not None:
                if fit_method == 'iterative_polynomial':
                    plt.plot(z_pos_for_polyfit, max_val_fitted,
                            label='Poly. Deg. {}'.format(current_guess['poly_deg']))
                elif fit_method == 'gaussian':
                    plt.plot(z_pos_for_polyfit, max_val_fitted,
                            label='Gauss Fit')
            #print("isclose: {}, {}, {}".format(new_focal_pos, focal_pos, np.isclose(new_focal_pos, focal_pos, rtol=1e-3, atol=1e-12)))
            if fit_method == 'iterative_polynomial':
                if np.isclose(new_focal_pos, focal_pos, atol=1e-9):
                    focal_plane_found = True
                    focal_pos = new_focal_pos
                else:
                    focal_pos = new_focal_pos
                    current_guess['poly_deg'] += 1
                    current_guess['focal_plane_index'] = focal_plane_nearest_index
            elif fit_method == 'gaussian':
                focal_pos = new_focal_pos
        focal_pos = np.round(focal_pos, 11)
        self.focal_plane_nearest_index = focal_plane_nearest_index
        if center_on_focal:
            self.z = self.z-focal_pos
            self.focal_plane = 0.
        elif center_on_nearest:
            nearest_z = self.z[self.focal_plane_nearest_index]
            self.z = self.z-nearest_z
            self.focal_plane = focal_pos-nearest_z
        else:
            self.focal_plane = focal_pos
        self.focal_plane_nearest_value = self.z[self.focal_plane_nearest_index]
        return focal_pos

    def _pre_process_guess(self, guess):
        """
        convert guess focal plane index 'center' to the central layer of stack

        Parameters
        ----------
        guess: dict
            the guess with field 'focal_plane_index'

        """
        if isinstance(guess['focal_plane_index'],str):
            if 'center' == guess['focal_plane_index']:
                guess['focal_plane_index'] = int((self.n_layers-1)/2.)
            else:
                raise ValueError("unknown focal plane starting guess with value:" +
                                 guess['focal_plane_index'])
        elif isinstance(guess['focal_plane_index'], (int, np.integer)):
            guess['focal_plane_index'] = guess['focal_plane_index']
        else:
            raise TypeError("focal plane starting guess must be integer or string")

    def _get_fit_param(self, fit_parameter):
        """
        Return a the fitting parameter per z plane

        Parameters
        ----------
        fit_parameter: str
            the parameter used for fitting

        """
        if fit_parameter == 'intensity':
            fit_param_per_plane = np.max(np.max(self.masked_data,axis=0), axis=0)
        elif fit_parameter == 'std_dev':
            fit_param_per_plane = self.std_per_layer()
        return fit_param_per_plane


    def fit_polynomial_to_data(self, guess, fit_parameter='intensity'):
        """
        Fit polynomial to the maximum value per plane.

        We seek the plane with the highest intensity value. At the same time,
        noisy data is assumed. We fit the max values per plane using a polynomial
        and take the maximum of this to avoid noise.

        Parameters
        ----------
        guess: str or int
            guess for the index nearest the focal plane, if "center" is given
            as input
        guess_area_width: int
            width in values of index to look around the focal plane guess
        polydef: int
            degree of polynomial fit to use
        """
        self._pre_process_guess(guess)
        guess_area_start = np.max([int(guess['focal_plane_index']-guess['area_width']/2),0])
        guess_area_end = np.min([int(guess['focal_plane_index']+guess['area_width']/2),self.z.size-1])
        fit_param_per_plane = self._get_fit_param(fit_parameter)
        fit_param_per_plane_sliced = fit_param_per_plane[guess_area_start:guess_area_end]
        z_pos_sliced = self.z[guess_area_start:guess_area_end]
        z_pos_for_polyfit = np.linspace( np.min(z_pos_sliced), np.max(z_pos_sliced), 10000)
        coeffs = np.polyfit(z_pos_sliced, fit_param_per_plane_sliced, guess['poly_deg'])
        fit_param_fitted = np.zeros(z_pos_for_polyfit.shape)
        for ii, coeff in enumerate(coeffs[::-1]):
            fit_param_fitted += coeff*np.power(z_pos_for_polyfit, ii)

        fitted_index_for_max = np.where(fit_param_fitted==np.max(fit_param_fitted))[0]
        focal_pos = z_pos_for_polyfit[fitted_index_for_max][0] # the z value for the maximum of the polyfit

        diff_to_focal_plane = np.abs(z_pos_sliced-focal_pos)
        focal_plane_index_nearest = guess_area_start + np.where(diff_to_focal_plane==np.min(diff_to_focal_plane))[0][0] # the index that is closest to the maximum

        return (focal_pos, focal_plane_index_nearest, z_pos_for_polyfit,
                fit_param_fitted)


    def fit_gaussian_to_data(self, guess, fit_parameter='intensity'):
        """
        Fit polynomial to the maximum value per plane.

        We seek the plane with the highest intensity value. At the same time,
        noisy data is assumed. We fit the max values per plane using a polynomial
        and take the maximum of this to avoid noise.

        Parameters
        ----------
        guess: str or int
            guess for the index nearest the focal plane, if "center" is given
            as input
        guess_area_width: int
            width in values of index to look around the focal plane guess
        polydef: int
            degree of polynomial fit to use
        """
        self._pre_process_guess(guess)
        guess_area_start = np.max([int(guess['focal_plane_index']-guess['area_width']/2),0])
        guess_area_end = np.min([int(guess['focal_plane_index']+guess['area_width']/2),self.z.size-1])
        fit_param_per_plane = self._get_fit_param(fit_parameter)
        fit_param_per_plane_sliced = fit_param_per_plane[guess_area_start:guess_area_end]
        z_pos_sliced = self.z[guess_area_start:guess_area_end]
        guess_std_dev = (z_pos_sliced[-1]-z_pos_sliced[0])*0.25
        start_vals = (np.max(fit_param_per_plane_sliced),
                      self.z[guess['focal_plane_index']+1], guess_std_dev)
        minimize_func = lambda x : utils.gaussian(x, z_pos_sliced)-fit_param_per_plane_sliced
        res = scipy.optimize.leastsq(minimize_func,
                                     x0=start_vals,
                                     full_output=False)
        z_pos_for_fit = np.linspace( np.min(z_pos_sliced), np.max(z_pos_sliced), 10000)
        fit_param_fitted = utils.gaussian(res[0], z_pos_for_fit)

        fitted_index_for_max = np.where(fit_param_fitted==np.max(fit_param_fitted))[0]
        focal_pos = z_pos_for_fit[fitted_index_for_max][0] # the z value for the maximum of the fit

        diff_to_focal_plane = np.abs(z_pos_sliced-focal_pos)
        focal_plane_index_nearest = guess_area_start + np.where(diff_to_focal_plane==np.min(diff_to_focal_plane))[0][0] # the index that is closest to the maximum
        return (focal_pos, focal_plane_index_nearest, z_pos_for_fit,
                fit_param_fitted)

    def max_afocal_range(self):
        left_range = np.abs(self.z[0]-self.focal_plane)
        right_range = np.abs(self.z[-1]-self.focal_plane)
        return np.min([left_range, right_range])


    def fit_parameters(self, basis='zernike', max_n_modes=10,
                       n_modes_per_layer=None, layers='all'):
        if basis == 'zernike_fringe' or basis == 'zernike':
            basis_func = basis_functions.zernike_fringe
            polar = True
            modes_start = 1
        elif basis == 'zernike_osa':
            basis_func = basis_functions.zernike_osa
            polar = True
            modes_start = 1
        elif basis == 'zernike_fringe_constrained':
            basis_func = basis_functions.zernike_fringe_constrained
            polar = True
            modes_start = 1
        elif basis == 'legendre':
            basis_func = basis_functions.legendre
            polar = False
            modes_start = 0
        else:
            raise ValueError("basis func: {} unknown".format(basis))

        if n_modes_per_layer is not None:
            max_n_modes = np.max(n_modes_per_layer)

        if layers == 'all':
            layers = range(self.n_layers)
        n_layers = len(layers)
        X, Y = self.get_cart_dimensions()[:2]
        if polar:
            X = X[:, :, 0]/self.mask_constraint
            Y = Y[:, :, 0]/self.mask_constraint
        else:
            X = X[:, :, 0]/self.mask_constraint[0]
            Y = Y[:, :, 0]/self.mask_constraint[1]
        image0 = self.masked_data[:,:,0]
        bd = BasisDecomposition(image0, basis_func, max_n_modes, modes_start,
                                X, Y, self.mask_region, polar)

        all_projected_coefficients = np.zeros((n_layers, max_n_modes))
        all_fitted_coefficients = np.zeros((n_layers, max_n_modes))
        all_std_devs = np.zeros((n_layers, max_n_modes))
        squared_residuals = np.zeros((n_layers))

        projected_data = ImageStack(np.zeros_like(self.data), self.x, self.y, self.z)
        fitted_data = ImageStack(np.zeros_like(self.data), self.x, self.y, self.z)
        #projected_slices = []
        #fitted_slices = []
        res = None
        for ilayer, layer in enumerate(layers):
            image = self.masked_data[:,:,layer]
            if n_modes_per_layer is not None:
                current_max_n_modes = n_modes_per_layer[ilayer]
            else:
                current_max_n_modes = max_n_modes
            bd.set_image(image)
            bd.set_current_max_n_modes(current_max_n_modes)
            projected_coefficients = bd.project_coefficients()[:current_max_n_modes]
            all_projected_coefficients[ilayer, :current_max_n_modes] = projected_coefficients
            projected_data.data[:,:, layer] = bd.image_from_coefficients(projected_coefficients)
            res = bd.fit_coefficients(projected_coefficients)
            all_fitted_coefficients[ilayer,:current_max_n_modes] = res[0]
            #res[1][np.isnan(res[1])] = 0.
            if res[1] is not None:
                diag_vals = np.squeeze(np.array([np.diag(res[1])]))
                diag_vals[diag_vals<0.] = 0.
                all_std_devs[ilayer, :current_max_n_modes] = np.sqrt(diag_vals+1e-10)
            fitted_data.data[:,:,layer] = bd.image_from_coefficients(res[0])
            squared_residuals[ilayer] = (res[2]['fvec']**2).sum()

        return (projected_data, all_projected_coefficients, res, projected_data,
                fitted_data, squared_residuals, all_fitted_coefficients, all_std_devs)

        #return coefficients

class BasisDecomposition():

    def __init__(self, image, basis_func, max_n_modes, modes_start, X, Y,
                 mask_region, polar):
        self.image = image
        self.max_n_modes = max_n_modes
        self.modes_start = modes_start
        self.current_max_n_modes = max_n_modes
        self.X = X
        self.Y = Y
        if polar:
            self.mask_shape = 'circular'
        else:
            self.mask_shape = 'rectangular'
        self.mask_region = mask_region
        self.basis_func = basis_func
        self.polar = polar
        self.init_basis()

    def init_basis(self):
        self.basis_functions = ImageStack.from_basis(self.basis_func,
                                                     self.max_n_modes,
                                                     self.modes_start,
                                                     self.X, self.Y,
                                                     polar=self.polar)
        if self.polar:
            constraint = 1.
        else:
            constraint = [1.0, 1.0]
        self.basis_functions.apply_mask(self.mask_shape, self.mask_region, constraint)

    def set_image(self, image):
        self.image = image

    def set_current_max_n_modes(self, n_modes):
        self.current_max_n_modes = n_modes

    def project_coefficients(self):
        return self.basis_functions.projection(self.image)

    def image_from_coefficients(self, coefficients):
        data = self.basis_functions.masked_data[:, :, :self.current_max_n_modes]
        im_stack = (coefficients*data)
        im_stack = im_stack.sum(axis=2)
        return im_stack

    def fit_coefficients(self, start_vals):
        minimize_func = lambda x : self.err_fun(x)
        jacob_fun = lambda x : self.err_fun_jac(x)
        res = scipy.optimize.leastsq(minimize_func,
                                     x0=start_vals,
                                     Dfun = jacob_fun,
                                     full_output=True)
        return res

    def err_fun(self, x):
        img = self.image_from_coefficients(x)
        diff = img - self.image
        return diff.ravel()

    def err_fun_jac(self, x):
        sliced_data = self.basis_functions.masked_data[:,:,:self.current_max_n_modes]
        return sliced_data.reshape(self.X.size, self.current_max_n_modes)
