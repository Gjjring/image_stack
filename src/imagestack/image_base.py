import numpy as np
from numpy import ma
from abc import ABC, abstractmethod
import scipy.optimize
import scipy.interpolate
import imagestack.statistics
from imagestack.mask import mask_from_data, copy_mask
#from imagestack.basis_decomposition import BasisDecomposition
from imagestack.utils import is_iterable
from imagestack.basis_functions import BasisFunctions
from imagestack import basis_functions
from imagestack.focal import FocalPlane
from imagestack.statistics import (InformationCriteria, information_criterion,
                                    residuals)

def _get_modes_from_basis(basis, n_modes):
    n_modes = int(n_modes)
    mode_start = basis_functions.mode_start(basis)
    return np.arange(mode_start, mode_start+n_modes, dtype=np.int64)


class ImageBase(ABC):

    """
    Abstract base class for N dimensional images

    Attributes
    ----------
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

    def __init__(self, data):
        """
        Parameters
        ----------
        data: np.ndarray
            The pixel data
        masked_data: np.ndarray
            The masked pixel data
        mask: image_base.mask.Mask
            object to easily apply masks to the data
        label: str
            label for this object
        """
        self._data = data
        self.masked_data = ma.array(data, copy=False)
        self.masked_data.mask = ma.make_mask_none(self.data.shape)
        self.mask = mask_from_data(data)
        self.label=""

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.masked_data = ma.array(new_data, copy=False)
        self.masked_data.mask = self.mask.current

    def add_noise(self, noise_level, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, noise_level, self.data.size).reshape(self.data.shape)
        self.data += noise

    def set_mask(self, shape, region, constraint, origin=None):
        self.mask.shape = shape
        self.mask.region = region
        self.mask.constraint = constraint
        if origin is not None:
            self.mask.origin = origin
        #mask = self.mask.generate_mask(self.get_cart_dimensions())
        self.apply_mask()

    def apply_mask(self, mask=None):
        if mask is not None:
            self.mask = copy_mask(mask)
        mask_array = self.mask.generate_mask(self.get_cart_dimensions())
        self.masked_data.mask = mask_array

    def update_data(self, new_data):
        self.data = new_data
        self.masked_data = ma.array(new_data, copy=False)
        self.masked_data.mask = self.mask.current

    def transform_data(self, scaling, constant):
        self.data = self.data*scaling + constant

    def normalise_flux(self, xy_scale=1.0):
        """
        normalise the data such that the flux is 1.
        """
        flux = self.flux(xy_scale=xy_scale)
        self.data = self.data/flux

    def normalise_range(self, edge_min=False, edge_max=False):
        """
        normalise the data such that the data lie between 0. and 1.
        """

        if edge_min:
            min_val = self.average_edge_data()
        else:
            min_val = np.amin(self.masked_data)
        if edge_max:
            max_val = self.average_edge_data()
        else:
            max_val = np.amax(self.masked_data)

        if np.isclose(np.abs(max_val-min_val), 0.):
            raise ValueError("cannot normalise range with equal maximum and" +
                             " minimum value")
        self.data = (self.data-min_val)/(max_val-min_val)

    def normalise_highest(self):
        """
        normalise the data such that the maximum value is 1.0
        """
        max_val = np.amax(self.masked_data)
        self.data /= max_val

    @abstractmethod
    def _check_image_compatible(self, other):
        """
        ensure that images are compatible for pixel comparison or assignment

        Paramters
        ---------
        other: subclass of ImageBase object
            the other image object to compare to
        """

    def pixel_comparison(self, other):
        """
        statistical data comparing the pixels.

        When comparing images, they must both have the same dimensionality, i.e.
        both by Image1D or Image2D objects. Additionally the unmasked pixels which
        will be compared must be associated with the same x and x/y positions for
        1D and 2D images respectively.

        Parameters
        ----------
        other: subclass of ImageBase
            a different image object instance to compare to
        """
        self._check_image_compatible(other)
        return residuals(self.masked_data.compressed(), other.masked_data.compressed())

    def normalise_reference_images(self, ref_upper, ref_lower):
        """
        normalise the data between an upper and lower reference image

        Parameters
        ----------
        ref_upper: subclass of ImageBase
            the upper reference image object instance
        ref_lower: subclass of ImageBase
            the lower reference image object instance
        """
        self._check_image_compatible(ref_upper)
        self._check_image_compatible(ref_lower)
        mask = self.mask.current
        new_data = ((self.masked_data.compressed()
                     -ref_lower.masked_data.compressed())
                    /(ref_upper.masked_data.compressed()
                      -ref_lower.masked_data.compressed()))
        self.data[np.logical_not(mask)] = new_data

    def normalise_reference_image_stacks(self, ref_upper, ref_lower):
        """
        normalise the data between an upper and lower reference image

        Parameters
        ----------
        ref_upper: ImageBase object
            the upper reference
        ref_lower: ImageBase object
            the lower reference
        """
        ref_image_upper = ref_upper.slice_z(z_value=self.z)
        ref_image_lower = ref_lower.slice_z(z_value=self.z)
        self._check_image_compatible(ref_image_upper)
        self._check_image_compatible(ref_image_lower)
        self.normalise_reference_images(ref_image_upper, ref_image_lower)

    def normalise_reference_values(self, upper, lower):
        """
        normalise the data between an upper and lower reference value

        Parameters
        ----------
        upper: float
            the upper value
        lower: float
            the lower value
        """
        new_data = (self.masked_data - lower)/(upper-lower)
        self.data[np.logical_not(self.mask.current)] = new_data.compressed()

    def std(self):
        """
        Standard deviation of the masked data
        """
        return np.std(self.masked_data)

    def max(self):
        """
        Maximum value of the masked data
        """
        return np.amax(self.masked_data)

    def min(self):
        """
        Minimum value of the masked data
        """
        return np.amin(self.masked_data)

    @abstractmethod
    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        raise NotImplementedError("method in abstract class should not be called")

    @abstractmethod
    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        raise NotImplementedError("method in abstract class should not be called")

    @classmethod
    @abstractmethod
    def from_basis(image1D, basis_func, n, x):
        raise NotImplementedError("method in abstract class should not be called")

    """
    @classmethod
    @abstractmethod
    def from_file(image1D, fpath):
        pass

    @abstractmethod
    def to_file(self, fpath):
        pass
    """

    @abstractmethod
    def central_value(self):
        raise NotImplementedError("method in abstract class should not be called")

    @abstractmethod
    def flux(self, xy_scale=1.0):
        raise NotImplementedError("method in abstract class should not be called")

    @abstractmethod
    def fit_basis(self, basis, n_modes):
        raise NotImplementedError("method in abstract class should not be called")



    @staticmethod
    def empty_fit_output(modes):
        """
        create struct with correct fields for basis fit output

        Parameters
        ----------
        modes: np.ndarray<N,>(int)
            the modes in the basis fit
        """
        n_modes = modes.size
        fit_output = {}
        fit_output['modes'] = modes
        fit_output['projected_coefficients'] = np.zeros((n_modes))
        fit_output['fitted_coefficients'] = np.zeros((n_modes))
        fit_output['std_devs'] = np.zeros((n_modes))
        fit_output['squared_residuals'] = np.zeros(1)
        return fit_output

    def _evalulate_info_criterion(self, basis, n_modes, criterion):
        """
        evaluate the information criteria for the given basis.

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use
        max_n_modes: int
            upper limit on number of basis coefficients
        criterion: InformationCriteria Enum
            the information criteria to be minimized
        """
        modes = _get_modes_from_basis(basis, n_modes)
        fit_output = self.fit_basis(basis, modes)
        residual = fit_output['squared_residuals']
        n_data_points = ma.count(self.masked_data)
        return information_criterion(residual, n_modes, n_data_points, criterion)

    def optimise_basis_size(self, basis, max_n_modes, min_n_modes=20,
                            criterion=InformationCriteria.BIC,
                            approximate=False):
        """
        Find the optimal number of basis coefficients for expanasion of the image

        We minimize the information criterion for a given set of basis functions
        to find how many functions are required to give a good fit to the data
        without overfitting.

        Note that this implementation seeks only a local minimum. To ensure that
        the global minimum is found, this routine can be called multiple times
        with different bounds.

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use
        max_n_modes: int
            upper limit on number of basis coefficients
        criterion: InformationCriteria Enum
            the information criteria to be minimized
        """
        obj_fun = lambda x : self._evalulate_info_criterion(basis, x,
                                                            criterion)
        #print("bounds: {}".format((min_n_modes, max_n_modes)))
        #return
        if approximate == False:
            res = scipy.optimize.minimize_scalar(obj_fun,
                                                 bounds=(min_n_modes, max_n_modes),
                                                 method='bounded',
                                                 options={'maxiter':20,
                                                          'xatol':0.5,
                                                          'disp':False})
            # attempted global minimizer but didnt seem to work:
            #res = scipy.optimize.shgo(obj_fun, bounds=[(min_n_modes, max_n_modes)],
            #                            options={'maxiter':100,
            #                                     'xatol':0.5,
            #                                     'disp':True})
            max_n_modes_required = int(res['x'])
        else:
            coarse_range = np.linspace(min_n_modes, max_n_modes, 5, dtype=np.int64)
            obj_values = np.zeros(coarse_range.size)
            #print("coarse range: {}".format(coarse_range))
            for ii, n_modes in enumerate(coarse_range):
                obj_values[ii] = obj_fun(n_modes)
                #print("obj value {}: {}".format(n_modes, obj_values[ii]))
            #print("obj_values: {}".format(obj_values))
            fine_range = np.arange(min_n_modes, max_n_modes+1, dtype=np.int64)
            interp = scipy.interpolate.interp1d(coarse_range, obj_values, kind='cubic')
            fine_values = interp(fine_range)
            min_index = np.where(np.isclose(fine_values,
                                            np.amin(fine_values)))
            max_n_modes_required = fine_range[min_index[0][0]]

        return max_n_modes_required


class ImageStackBase(ABC):

    """
    Abstract base class stacks for N dimensional images

    Attributes
    ----------
    data: (N,M,K) np.ndarray
        The pixel data in the layers
    masked_array: (N,M,K) np.ma.maskedarray
        The masked pixel data
    x: (N,) np.ndarray
        The positions of the pixels in the x direction
    y: (M,) np.ndarray
        The positions of the pixels in the y direction
    z: (K,) np.ndarray
        The positions of the images in the z direction
    n_layers: int
        The number of images layers
    mask_shape: str
        The shape of the mask for data parameter fitting
    mask_region: str
        The region of the mask for data parameter fitting
    mask_constraint: float or (2,) tuple
        constraint for the mask for data parameter fitting
    label: str
        label used for plotting
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data: np.ndarray
            The pixel data
        masked_data: np.ndarray
            The masked pixel data
        mask: image_base.mask.Mask
            object to easily apply masks to the data
        label: str
            label for this object
        """
        self._data = data
        self.masked_data = ma.array(data, copy=False)
        self.masked_data.mask = ma.make_mask_none(data.shape)
        self.mask = mask_from_data(data[..., 0])
        self.focal_plane = None
        self.label= ""

    def add_noise(self, noise_level, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, noise_level, self.data.size).reshape(self.data.shape)
        self.data += noise

    def set_mask(self, shape, region, constraint, origin=None):
        self.mask.shape = shape
        self.mask.region = region
        self.mask.constraint = constraint
        if origin is not None:
            self.mask.origin = origin
        mask = self.mask.generate_mask(self.get_cart_dimensions())
        self.apply_mask()



    def apply_mask(self, mask=None):
        if mask is not None:
            self.mask = copy_mask(mask)
        mask_array = self.mask.generate_mask(self.get_cart_dimensions())
        for z_index in range(self.z.size):
            self.masked_data.mask[..., z_index] = mask_array

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.masked_data = ma.array(new_data, copy=False)
        self.masked_data.mask = ma.make_mask_none(new_data.shape)
        for z_index in range(new_data.shape[-1]):
            self.masked_data.mask[..., z_index] = self.mask.current

    def transform_data(self, scaling, constant):
        self.data = self.data*scaling + constant

    def normalise_flux(self, xy_scale=1.0):
        """
        normalise the data such that the flux is 1.
        """
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            flux = image.flux(xy_scale=xy_scale)
            self.data[..., layer] /= flux

    def normalise_range(self, layer_by_layer=True, edge_min=False, edge_max=False):
        """
        normalise the data such that the data lie between 0. and 1.
        """
        if layer_by_layer:
            for layer in range(self.n_layers):
                image = self.slice_z(z_index=layer)

                if edge_min:
                    min_val = image.average_edge_data()
                else:
                    min_val = np.min(image.masked_data)

                if edge_max:
                    max_val = image.average_edge_data()
                else:
                    max_val = np.max(image.masked_data)


                if np.isclose(np.abs(max_val-min_val), 0.):
                    raise ValueError("cannot normalise range with equal maximum and" +
                                     " minimum value")
                self.data[..., layer] = (self.data[..., layer]-min_val)/(max_val-min_val)
        else:

            if edge_min:
                min_val = np.amin(self.average_edge_data())
            else:
                min_val = np.min(self.masked_data)
            if edge_max:
                max_val = np.amin(self.average_edge_data())
            else:
                max_val = np.max(self.masked_data)
            self.data = (self.data-min_val)/(max_val-min_val)

    def normalise_highest(self):
        """
        normalise the data such that the maximum value is 1.0
        """
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            max_val = np.max(image.masked_data)
            self.data[..., layer] /= max_val

    def pixel_comparison(self, other):
        """
        statistical data comparing the pixels.

        When comparing images, they must both have the same dimensionality, i.e.
        both by Image1D or Image2D objects. Additionally the unmasked pixels which
        will be compared must be associated with the same x and x/y positions for
        1D and 2D images respectively.

        Parameters
        ----------
        other: subclass of ImageBase
            a different image object instance to compare to
        """
        self._check_stack_compatible(other)
        return residuals(self.masked_data.compressed(), other.masked_data.compressed())

    def normalise_reference_values(self, upper, lower):
        """
        normalise the data between an upper and lower reference value

        Parameters
        ----------
        upper: float
            the upper value
        lower: float
            the lower value
        """
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            image.normalise_reference_values(upper, lower)
            self.data[..., layer] = image.data

    def normalise_reference_images(self, ref_upper, ref_lower):
        """
        normalise the data between an upper and lower reference image

        Parameters
        ----------
        ref_upper: ImageBase object
            the upper reference
        ref_lower: ImageBase object
            the lower reference
        """
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            image.normalise_reference_images(ref_upper, ref_lower)
            self.data[..., layer] = image.data

    def normalise_reference_image_stacks(self, ref_upper, ref_lower):
        """
        normalise the data between an upper and lower reference image

        Parameters
        ----------
        ref_upper: ImageStackBase object
            the upper reference
        ref_lower: ImageStackBase object
            the lower reference
        """
        self._check_stack_compatible(ref_upper)
        self._check_stack_compatible(ref_upper)
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            ref_image_upper = ref_upper.slice_z(z_index=layer)
            ref_image_lower = ref_lower.slice_z(z_index=layer)
            image.normalise_reference_images(ref_image_upper, ref_image_lower)
            self.data[..., layer] = image.data


    def _check_stack_compatible(self, other):
        if not isinstance(self, type(other)):
            raise ValueError("cannot compare image stack of type {}".format(type(self)) +
                             " to image of type {}".format(type(other)))

        if not self.z.size == other.z.size:
            raise ValueError("z positions are of different size " +
                             "image_stack1 size: {} ".format(self.z.size) +
                             "image_stack2 size: {}".format(other.z.size))

        if not np.all(np.isclose(self.z,other.z)):
            raise ValueError("cannot compare image_stacks with different z positions")

    def _slice_z(self, z_index=None, z_value=None):
        if z_index is None and z_value is None:
            raise ValueError("for z slice, either z_index or z_value must not be None")
        if z_index is not None and z_value is not None:
            raise ValueError("for z slice, only one of z_index or z_value may not be None")
        if z_value is not None:
            if not is_iterable(z_value):
                z_index = np.where(np.isclose(self.z, z_value))[0]
            else:
                z_index = []
                for value in z_value:
                    slice_index = np.where(np.isclose(self.z, value))[0]
                    if len(slice_index) == 0:
                        raise ValueError("cannot slice for z position {}".format(value) +
                                         ", it is not in the image stack")
                    z_index.append(slice_index[0])

            if len(z_index) == 0:
                raise ValueError("cannot slice for z position {}, it is not in the image stack".format(z_value))
        else:
            z_index = [z_index]
        if len(z_index)>1:
            is_sequence = True
        else:
            is_sequence = False
        sliced_data = np.squeeze(self.data[..., z_index])
        sliced_z = self.z[z_index]
        return sliced_data, sliced_z, is_sequence

    def std(self):
        """
        Standard deviation of the masked data for each layer
        """
        std_devs = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            std_devs[layer] = np.std(self.masked_data[..., layer])
        return std_devs

    def max(self):
        """
        Maximum value of the masked data
        """
        max_vals = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            max_vals[layer] = np.max(self.masked_data[..., layer])
        return max_vals

    def min(self):
        """
        Minimum value of the masked data
        """
        min_vals = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            min_vals[layer] = np.min(self.masked_data[..., layer])
        return min_vals

    def flux(self):
        """
        Minimum value of the masked data
        """
        flux_vals = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            image = self.slice_z(z_index = layer)
            flux_vals[layer] = image.flux()
        return flux_vals

    def optimise_basis_size(self, basis, max_n_modes, min_n_modes=20,
                            criterion=InformationCriteria.BIC,
                            approximate=False):
        """
        Find the optimal number of basis coefficients for expanasion of the image

        We minimize the information criterion for a given set of basis functions
         to find how many functions are required to give a good fit to the data
        without overfitting.

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use
        max_n_modes: int
            upper limit on number of basis coefficients
        criterion: InformationCriteria Enum
            the information criteria to be minimized
        """
        all_required_modes = np.zeros(self.n_layers, dtype=np.int64)
        for layer in range(self.n_layers):
            image = self.slice_z(layer)
            if layer == 0:
                modes = _get_modes_from_basis(basis, min_n_modes)
                bd = image.get_basis_decomp(basis, modes)
            else:
                bd.reset_projection_cache()
                image.basis_decomp = bd
            required_n_modes = image.optimise_basis_size(basis, max_n_modes,
                                                         min_n_modes=min_n_modes,
                                                         criterion=criterion,
                                                         approximate=approximate)
            all_required_modes[layer] = required_n_modes
        return all_required_modes

    def determine_focal_plane(self, model='gaussian', measure='std_dev'):
        z = self.z
        if measure == 'std_dev':
            data = self.std()
        elif measure == 'max_val':
            data = self.max()
        else:
            raise ValueError("unknown focal plane fitting measure: {}".format(measure))
        focal_fitter = FocalPlane(z, data, model=model)
        fit_window_width = focal_fitter.optimize_fit_window()
        focus_estimate = focal_fitter.fit(fit_window=fit_window_width,
                                          fit_center=focal_fitter.current_focus_estimate)[0]

        return focus_estimate

    def fit_basis(self, basis, modes):
        """
        fit modes of a basis to the masked images in the stack

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use for fitting
        modes: sequence of np.ndarray<N,>(int) values
            the modes of the basis function per layer
        """
        if not len(modes) == self.z.size:
            raise ValueError("n_modes must have same length as layers in stack")
        all_fit_outputs = []
        max_n_modes = 0
        for layer_modes in modes:
            max_n_modes = np.max([max_n_modes, np.amax(layer_modes)])
        for layer in range(self.n_layers):
            image = self.slice_z(z_index=layer)
            if layer == 0:
                bd = image.get_basis_decomp(basis, modes[layer])
                image.basis_decomp = bd
            else:
                image.basis_decomp = bd
            fit_output = image.fit_basis(basis, modes[layer])
            all_fit_outputs.append(fit_output)
        return all_fit_outputs

    def max_afocal_range(self):
        return np.min([np.abs(np.max(self.z)), np.abs(np.min(self.z))])

    def center_on_focus(self):
        self.z -= self.focal_plane
        self.focal_plane = 0.

    def center_on_nearest_to_focus(self):
        dif_to_focus = np.abs(self.z - self.focal_plane)
        z_index = np.where(dif_to_focus == np.min(dif_to_focus))[0][0]
        close_to_focus = self.z[z_index]
        self.z -= close_to_focus
        self.focal_plane = self.focal_plane-close_to_focus

    @abstractmethod
    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        raise NotImplementedError("method in abstract class should not be called")

    @abstractmethod
    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        raise NotImplementedError("method in abstract class should not be called")

    @classmethod
    @abstractmethod
    def from_basis(image1D, basis_func, n, x):
        """
        returns a line image of a basis function of order n evaluated on x

        Parameters
        ----------
        basis_func: function handle
            the basis function to use
        n: int
            the order of the basis function
        x: nd.array<float>(N,)
            the positions over which to evaluate the objective function
        """
        raise NotImplementedError("method in abstract class should not be called")

    @abstractmethod
    def slice_z(self, z_index=None, z_value=None):
        """
        slice the stack and return an image of the appropriate dimension.
        """
        raise NotImplementedError("method in abstract class should not be called")

    """
    @classmethod
    @abstractmethod
    def from_file(image1D, fpath):
        pass

    @abstractmethod
    def to_file(self, fpath):
        pass
    """

    @abstractmethod
    def central_value(self):
        raise NotImplementedError("method in abstract class should not be called")



    # @abstractmethod
    # def flux(self, xy_scale=1.0):
    #     raise NotImplementedError("method in abstract class should not be called")


    # @abstractmethod
    # def fit_basis(self, basis, n_modes):
    #     raise NotImplementedError("method in abstract class should not be called")
