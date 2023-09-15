import numpy as np
from numpy import ma
from imagestack.image_base import ImageBase, ImageStackBase
from  imagestack import basis_functions
from imagestack.mask import Mask1D
import scipy.optimize
from imagestack.statistics import residuals
#from imagestack.basis_decomposition import BasisDecomposition
class Image1D(ImageBase):

    """
    1D pixel image with z position

    Attributes
    ---------
    data: (N,) np.ndarray
        The pixel data
    masked_array: (N,) np.ma.maskedarray
        The masked pixel data
    x: (N,) np.ndarray
        The positions of the pixels in the x direction
    z: float
        The z position of the image
    """

    def __init__(self, data, x, z):
        """
        Parameters
        ----------
        data: (N,) np.ndarray
            The pixel data
        x: (N,) np.ndarray
            The positions of the pixels in the x direction
        z: float
            The position of the image in the z direction
        """
        super().__init__(data)
        self.x = x
        self.z = z

    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        return self.x

    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        raise ValueError("Image1D does not have cylindrical dimensions")

    def xlim(self):
        if self.mask.current is not None:
            x = self.x[np.logical_not(self.mask.current)]
        else:
            x = self.x
        min_x = np.min(x)
        max_x = np.max(x)
        return (min_x, max_x)

    def average_edge_data(self):
        complement_mask = Mask1D.from_mask(self.mask, complement=True)
        complement_mask.constraint = complement_mask.constraint*0.95
        x = self.x
        complement_mask_array = complement_mask.generate_mask(x)
        complement_masked_data = ma.array(self.data, copy=True)
        complement_masked_data.mask = np.logical_or(complement_mask_array, self.mask.current)
        return complement_masked_data.mean()

    @classmethod
    def from_basis(image1D, basis, n, x):
        """
        returns a Image1D of a basis function of order n evaluated on x

        Parameters
        ----------
        basis_func: function handle
            the basis function to use
        n: int
            the order of the basis function
        x: nd.array<float>(N,)
            the positions over which to evaluate the objective function
        """
        data = np.zeros(x.size)
        dimension1 = x
        len_x = x.size

        data = basis_functions.function(basis)(n, dimension1)
        return image1D(data, x, float(n))

    @classmethod
    def from_file(image1D, fpath):
        with np.load(fpath, allow_pickle=True) as file_data:
            required_fields = ['data', 'x', 'z']
            for field in required_fields:
                if field not in file_data:
                    raise KeyError("required field : {}".format(field) +
                                   " is missing from npz file")
            img_stack = image1D(file_data['data'], file_data['x'],
                                      file_data['z'])
            if 'mask_shape' in file_data:
                if 'mask_origin' in file_data:
                    origin = file_data['mask_origin']
                else:
                    origin = None
                img_stack.set_mask(file_data['mask_shape'],
                                   file_data['mask_region'],
                                   file_data['mask_constraint'],
                                   origin=origin)
        return img_stack


    def to_file(self, fpath):
        if self.mask.current is not None:
            np.savez(fpath, x=self.x, z=self.z,
                     data=self.data, mask_shape=self.mask.shape,
                     mask_region=self.mask.region,
                     mask_constraint=self.mask.constraint)
        else:
            np.savez(fpath, x=self.x, z=self.z,
                     data=self.data)

    def central_value(self):
        """
        returns the value at the center of the image
        """

        x_length = self.x.size
        x_mid = int((x_length-1)/2)
        return self.masked_data[x_mid]

    def flux(self, xy_scale=1.0):
        """
        integrate the intensity over the x dimension

        Parameters
        ----------
        xy_scale: float
            scaling factor for the lateral (x) dimension
        """
        flux = np.trapz(self.masked_data, self.x*xy_scale)
        return flux

    def abs_dif(self, other):
        """
        absolute difference of pixel value between two images

        Parameters
        ----------
        other: Image1D
            the other image to compare the pixel values with
        """
        self._check_image_compatible(other)
        new_data = np.zeros(self.data.shape)
        self_masked_data = self.masked_data.compressed()
        other_masked_data = other.masked_data.compressed()
        new_data[np.logical_not(self.mask.current)] = np.abs(self_masked_data-other_masked_data)
        image = Image1D(new_data, self.x, self.z)
        image.apply_mask(self.mask)
        return image

    def dif(self, other):
        """
        signed difference of pixel value between two images

        Parameters
        ----------
        other: Image1D
            the other image to compare the pixel values with
        """
        self._check_image_compatible(other)
        new_data = np.zeros(self.data.shape)
        self_masked_data = self.masked_data.compressed()
        other_masked_data = other.masked_data.compressed()
        new_data[np.logical_not(self.mask.current)] = self_masked_data-other_masked_data
        image = Image1D(new_data, self.x, self.z)
        image.apply_mask(self.mask)
        return image

    def _basis_projection(self, basis_decomp, fit_output):
        """
        find the basis coefficients for a projection of the masked data

        Parameters
        ----------
        basis_decomp: BasisDecomposition1D
            ImageStack of basis functions
        fit_output: dict
            dict for holding fit information
        """
        projected_data = np.zeros_like(self.data)
        projected_coefficients = basis_decomp.projection(self.masked_data)
        fit_output['projected_coefficients'][...] = projected_coefficients
        projected_data[...] = basis_decomp.image_from_coefficients(projected_coefficients)
        fit_output['projected_image'] = Image1D(projected_data, self.x, self.z)
        fit_output['projected_image'].apply_mask(self.mask)

    def _optimize_projection(self, basis_decomp, fit_output):
        """
        find the optimized coefficients for a projection of the masked data

        Parameters
        ----------
        basis_decomp: BasisDecomposition1D
            ImageStack of basis functions
        fit_output: dict
            dict for holding fit information
        """
        fitted_data = np.zeros_like(self.data)
        res = basis_decomp._fit_coefficients(self.masked_data,
                                            fit_output['projected_coefficients'])
        fit_output['fitted_coefficients'][...] = res[0]
        std_floor = 1e-8
        if res[1] is not None:
            diag_vals = np.squeeze(np.array([np.diag(res[1])]))
            diag_vals[diag_vals<std_floor] = np.abs(res[0][diag_vals<std_floor])*1e-2
            fit_output['std_devs'][...] = np.sqrt(diag_vals)
        fitted_data[...] = basis_decomp.image_from_coefficients(res[0])
        fit_output['squared_residuals'] = (res[2]['fvec']**2).sum()
        fit_output['fitted_image'] = Image1D(fitted_data, self.x, self.z)
        fit_output['fitted_image'].apply_mask(self.mask)

    def fit_basis(self, basis, modes):
        """
        fit n modes of a basis to the masked image

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use for fitting
        n_modes: int
            the number of basis functions to use
        """
        is_polar = basis_functions.is_polar(basis)
        #mode_start = basis_functions.mode_start(basis)
        #modes = np.arange(mode_start, mode_start+n_modes, dtype=np.int64)
        if is_polar:
            raise ValueError("polar basis functions incompatible with 1D image")

        bd = BasisDecomposition1D(basis, modes, self.x, self.mask)
        fit_output = ImageBase.empty_fit_output(modes)
        projected_coefficients = self._basis_projection(bd, fit_output)
        self._optimize_projection(bd, fit_output)
        return fit_output

    def _check_image_compatible(self, other):
        """
        ensure that images are compatible for pixel comparison or assignment

        Paramters
        ---------
        other: Image1D object
            the other image object to compare to
        """
        if not isinstance(self,type(other)):
            raise ValueError("cannot compare image of type {}".format(type(self)) +
                             " to image of type {}".format(type(other)))

        self_masked_x = ma.array(self.x, mask=self.mask.current)
        other_masked_x = ma.array(other.x, mask=other.mask.current)
        if not self_masked_x.count() == other_masked_x.count():
            raise ValueError("masked x positions are of different size " +
                             "image1 size: {} ".format(self_masked_x.count()) +
                             "image2 size: {}".format(other_masked_x.count()))

        if not np.all(np.isclose(self_masked_x.compressed(),other_masked_x.compressed())):
            raise ValueError("cannot compare images with different unmasked x positions")


class ImageStack1D(ImageStackBase):

    def __init__(self, data, x, z):
        """
        Parameters
        ----------
        data: (N,) np.ndarray
            The pixel data
        x: (N,) np.ndarray
            The positions of the pixels in the x direction
        z: (N,) np.ndarray
            The position of the image in the z direction
        """
        super().__init__(data)
        self.x = x
        self.z = z
        self.n_layers = z.size

    def slice_z(self, z_index=None, z_value=None):
        sliced_data, sliced_z, is_sequence = self._slice_z(z_index=z_index, z_value=z_value)
        if is_sequence:
            sliced_image = ImageStack1D(sliced_data, self.x, sliced_z)
        else:
            sliced_image = Image1D(sliced_data, self.x, sliced_z)
        sliced_image.apply_mask(mask=self.mask)
        sliced_image.label = self.label
        return sliced_image

    def get_cart_dimensions(self):
        """
        Create a meshgrid of the cartesian position vectors
        """
        return self.x

    def get_cyl_dimensions(self):
        """
        Create a meshgrid of the cylindrical position vectors
        """
        raise ValueError("Image1D does not have cylindrical dimensions")

    @classmethod
    def from_basis(image_stack1D, basis, modes, x):
        """
        returns a ImageStack1D of a basis function of various orders evaluated on x

        Parameters
        ----------
        basis_func: function handle
            the basis function to use
        n_min: int
            the first order of the basis function
        n_max: int
            the last order of the basis function
        x: nd.array<float>(N,)
            the positions over which to evaluate the basis functions
        """
        n_modes = modes.size
        data = np.zeros((x.size, n_modes))
        dimension1 = x
        len_x = x.size
        #modes = np.array(list(range(n_min, n_max+1)), dtype=np.int64)
        basis_func = basis_functions.function(basis)
        for ii, mode in enumerate(modes):
            data[:, ii] = basis_func(mode, dimension1)
        return image_stack1D(data, x, modes.astype(np.double))

    @classmethod
    def from_image_list(image_stack1D, images):
        """
        returns an ImageStack1D of by concatenating a list of images into one
        stack

        Parameters
        ----------
        images: sequence of Image1D objects
            the images to concatenate
        """
        x = images[0].x
        z_vals = np.zeros(len(images))
        data = np.zeros((x.size, z_vals.size))
        for iz in range(len(images)):
            image = images[iz]
            z_vals[iz] = image.z
            data[:, iz] = image.data
        image_stack1D = ImageStack1D(data, x, z_vals)
        image_stack1D.apply_mask(images[0].mask)
        return image_stack1D



    @classmethod
    def from_file(image_stack1D, fpath):
        with np.load(fpath, allow_pickle=True) as file_data:
            required_fields = ['data', 'x', 'z']
            for field in required_fields:
                if field not in file_data:
                    raise KeyError("required field : {}".format(field) +
                                   " is missing from npz file")
            img_stack = image_stack1D(file_data['data'], file_data['x'],
                                      file_data['z'])
            if 'mask_shape' in file_data:
                if 'mask_origin' in file_data:
                    origin = file_data['mask_origin']
                else:
                    origin = None
                img_stack.set_mask(file_data['mask_shape'],
                                   file_data['mask_region'],
                                   file_data['mask_constraint'],
                                   origin=origin)
        return img_stack


    def to_file(self, fpath):
        if self.mask.current is not None:
            np.savez(fpath, x=self.x, z=self.z,
                     data=self.data, mask_shape=self.mask.shape,
                     mask_region=self.mask.region,
                     mask_constraint=self.mask.constraint)
        else:
            np.savez(fpath, x=self.x, z=self.z,
                     data=self.data)

    def central_value(self):
        x_length = self.x.size
        x_mid = int((x_length-1)/2)
        return self.masked_data[x_mid, :]

    def projection(self, masked_1d_data):
        n_unmasked = ma.count(masked_1d_data)
        coefficients = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            basis_func = self.masked_data[..., layer]
            coef = ma.sum(masked_1d_data*basis_func)/n_unmasked
            coefficients[layer] = coef
        return coefficients

    def average_edge_data(self):
        """
        return average value of masked data close to mask edge.

        Returns
        -------
        np.ndarray<N,>(float)
          - the average value of the edge data in the N layers
        """
        edge_average = np.zeros(self.n_layers)
        for layer in range(self.n_layers):
            sliced_image = self.slice_z(z_index=layer)
            edge_average[layer] = sliced_image.average_edge_data()
        return edge_average


class BasisDecomposition1D(ImageStack1D):
    """
    Fit a set of basis functions to 1D image data
    """

    def __init__(self, basis_func, modes,
                 x, mask):
        """
        Parameters
        ----------
        basis_func: BasisFunctions Enum
            The set of functions to use for the decomposition
        modes: nd.array<int>(M,)
            the indices of the modes to use in the decomposition
        x: nd.array<float>(N,)
            the positions over which to evaluate the basis functions
        mask: imagestack.mask.Mask object
            mask for the data
        """
        self.basis = basis_func
        self.modes = modes
        mask = Mask1D.from_mask(mask)
        x = BasisDecomposition1D.normalise_dimensions(x, mask)
        data = self.init_basis(x)
        if not isinstance(modes, np.ndarray):
            raise AttributeError("modes must by numpy.ndarray")
        super().__init__(data, x, modes.astype(float))
        self.apply_mask(mask)


    @staticmethod
    def normalise_dimensions(x, mask):
        normed_x =  x/mask.constraint
        mask.constraint = 1.0
        return normed_x

    def init_basis(self, x):
        return ImageStack1D.from_basis(self.basis, self.modes, x).data

    def image_from_coefficients(self, coefficients):
        stacked_image_data = (coefficients*self.masked_data)
        image_data = stacked_image_data.sum(axis=1)
        return image_data

    def _fit_coefficients(self, image, start_vals):
        self.image_data = image
        minimize_func = lambda x : self.err_fun(x)
        jacob_fun = lambda x : self.err_fun_jac(x)
        res = scipy.optimize.leastsq(minimize_func,
                                     x0=start_vals,
                                     Dfun = jacob_fun,
                                     full_output=True)
        return res

    def err_fun(self, x):
        img = self.image_from_coefficients(x)
        diff = img - self.image_data
        return diff.ravel()

    def err_fun_jac(self, x):
        sliced_data = self.masked_data
        return sliced_data
