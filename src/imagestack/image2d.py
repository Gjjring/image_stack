import numpy as np
from numpy import ma
from imagestack.image_base import ImageBase, ImageStackBase
from imagestack.integrator import Integrator2D
from imagestack.focal import CenteredGaussianModel
from imagestack import basis_functions
from imagestack.image1d import Image1D, ImageStack1D
from imagestack.mask import Mask2D
import scipy.optimize
from scipy import ndimage
from imagestack.statistics import residuals
from copy import copy


class Image2D(ImageBase):

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
    """

    def __init__(self, data, x, y, z):
        """
        Parameters
        ----------
        data: (N,M) np.ndarray
            The pixel data
        x: (N,) np.ndarray
            The positions of the pixels in the x direction
        y: (M,) np.ndarray
            The positions of the pixels in the y direction
        z: float
            The position of the image in the z direction
        """
        super().__init__(data)
        self.x = x
        self.y = y
        self.z = z
        self.basis_decomp = None

    @classmethod
    def from_image(image2d, other_image):
        return image2d(copy(other_image.data), copy(other_image.x),
                       copy(other_image.y), copy(other_image.z))


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

    def xlim(self):
        X, Y = self.get_cart_dimensions()
        if self.mask.current is not None:
            x = np.unique(X[np.logical_not(self.mask.current)])
        else:
            x = np.unique(X)
        min_x = np.min(x)
        max_x = np.max(x)
        return (min_x, max_x)

    def ylim(self):
        X, Y = self.get_cart_dimensions()
        if self.mask.current is not None:
            y = np.unique(Y[np.logical_not(self.mask.current)])
        else:
            y = np.unique(Y)
        min_y = np.min(y)
        max_y = np.max(y)
        return (min_y, max_y)

    @classmethod
    def from_basis(image2D, basis, n, x, y):
        """
        returns a 2D image of a basis function of order n evaluated on x/y

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
        polar = basis_functions.is_polar(basis)
        X, Y = np.meshgrid(x, y, indexing='ij')
        if polar:
            R = np.sqrt(X**2 + Y**2)
            PHI = np.arctan2(Y,X)
            dimension1 = R
            dimension2 = PHI
        else:
            dimension1 = X
            dimension2 = Y
        modal_func = basis_functions.function(basis)(n, dimension1, dimension2)
        return image2D(modal_func, x, y, float(n))

    @classmethod
    def from_file(image2D, fpath):
        with np.load(fpath, allow_pickle=True) as file_data:
            required_fields = ['data', 'x', 'y', 'z']
            for field in required_fields:
                if field not in file_data:
                    raise KeyError("required field : {}".format(field) +
                                   " is missing from npz file")

            img_stack = image2D(file_data['data'], file_data['x'],
                                      file_data['y'], file_data['z'])
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
            np.savez(fpath, x=self.x, y=self.y, z=self.z,
                     data=self.data, mask_shape=self.mask.shape,
                     mask_region=self.mask.region,
                     mask_constraint=self.mask.constraint)
        else:
            np.savez(fpath, x=self.x, y=self.y, z=self.z,
                     data=self.data)


    def central_value(self):
        """
        returns the value at the center of the image
        """
        x_length = self.x.size
        y_length = self.y.size
        x_mid = int((x_length-1)/2)
        y_mid = int((y_length-1)/2)
        return self.masked_data[x_mid, y_mid]

    def average_over_dimension(self, dimension='y', use_masked=False):
        """
        average data values over a lateral dimension and return Image1D

        Parameters
        ----------
        dimension: string |'x'|'y'|
            the dimension to average over
        use_masked: bool
            average the masked data
        """
        if dimension == 'x':
            if use_masked:
                new_data = self.masked_data.mean(axis=0)
                if self.mask.polar:
                    constraint = self.mask.constraint
                else:
                    constraint = self.mask.constraint[1]
            else:
                new_data = self.data.mean(axis=0)
            new_x = self.y
            new_y = np.array([0.])
            origin = self.mask.origin[0]
        elif dimension == 'y':
            if use_masked:
                new_data = self.masked_data.mean(axis=1)
                print(self.mask.shape)
                if self.mask.shape=='circular':
                    constraint = self.mask.constraint
                else:
                    constraint = self.mask.constraint[0]
            else:
                new_data = self.data.mean(axis=1)
            new_x = self.x
            new_y = np.array([0.])
            origin = self.mask.origin[1]
        else:
            raise ValueError("dimension must by either x or y", +
                             " ,value was: {}".format(dimension))
        #print("new_data.shape: {}".format(new_data.shape))
        image1d = Image1D(new_data, new_x, self.z)
        if use_masked:
            image1d.set_mask("window", self.mask.region, constraint, origin=origin)
        return image1d

    def slice_dimension(self, dimension='y', position=0., use_masked=False):
        """
        average data values over a lateral dimension and return Image1D

        Parameters
        ----------
        dimension: string |'x'|'y'|
            the dimension to average over
        position: float
            the position along the dimension at which to slice
        use_masked: bool
            average the masked data
        """
        if dimension == 'x':
            slice_index = np.where(np.isclose(self.x, position))[0][0]
            if use_masked:
                new_data = self.masked_data[slice_index, :]
            else:
                new_data = self.data[slice_index, :]
            new_x = self.y
            new_y = np.array([position])
        elif dimension == 'y':
            slice_index = np.where(np.isclose(self.y, position))[0][0]
            if use_masked:
                new_data = self.masked_data[:, slice_index]
            else:
                new_data = self.data[:, slice_index]
            new_x = self.x
            new_y = np.array([position])
        else:
            raise ValueError("dimension must by either x or y", +
                             " ,value was: {}".format(dimension))
        #print("new_data.shape: {}".format(new_data.shape))
        image = Image1D(new_data, new_x, self.z)
        if self.mask.shape =='circular':
            constraint = self.mask.constraint
        elif self.mask.shape =='rectangular':
            if dimension == 'x':
                constraint = self.mask.constraint[1]
            elif dimension == 'y':
                constraint = self.mask_constraint[0]
        else:
            constraint = None
        if constraint is not None:
            image.set_mask("window", self.mask.region, constraint)
        return image



    def flux(self, xy_scale=1.0):
        """
        integrate the masked data over the x & y dimensions

        Parameters
        ----------
        xy_scale: float
            scaling factor for the lateral (x,y) dimensions
        """
        X, Y = self.get_cart_dimensions()[:2]
        if self.x.size > 1 and self.y.size > 1:
            if self.mask.shape == 'circular':
                xy = xy_scale*np.vstack([X.flatten(), Y.flatten()]).T
                int2D = Integrator2D(xy, self.masked_data.flatten())
                int_val = int2D.integrate_circle(self.mask.constraint*xy_scale)
            elif self.mask.shape == 'rectangular' or self.mask.shape==None:
                int_val = np.trapz(np.trapz(self.masked_data,
                                            xy_scale*self.x, axis=0),
                                   xy_scale*self.y, axis=0)
        else:
            raise ValueError("cannot get flux for lateral dimensions of "+
                             "x,y : {},{}".format(self.x.size, self.y.size))
        flux = int_val
        return flux

    def abs_dif(self, other):
        """
        absolute difference of pixel value between two images

        Parameters
        ----------
        other: Image2D
            the other image to compare the pixel values with
        """
        self._check_image_compatible(other)
        new_data = np.zeros(self.data.shape)
        self_masked_data = self.masked_data.compressed()
        other_masked_data = other.masked_data.compressed()
        new_data[np.logical_not(self.mask.current)] = np.abs(self_masked_data-other_masked_data)
        image = Image2D(new_data, self.x, self.y, self.z)
        image.apply_mask(self.mask)
        return image

    def dif(self, other):
        """
        signed difference of pixel value between two images

        Parameters
        ----------
        other: Image2D
            the other image to compare the pixel values with
        """
        self._check_image_compatible(other)
        new_data = np.zeros(self.data.shape)
        self_masked_data = self.masked_data.compressed()
        other_masked_data = other.masked_data.compressed()
        new_data[np.logical_not(self.mask.current)] = self_masked_data-other_masked_data
        image = Image2D(new_data, self.x, self.y, self.z)
        image.apply_mask(self.mask)
        return image

    def _basis_projection(self, basis_decomp, fit_output):
        """
        find the basis coefficients for a projection of the masked data

        Parameters
        ----------
        basis_decomp: BasisDecomposition2D
            ImageStack of basis functions
        fit_output: dict
            dict for holding fit information
        """
        projected_data = np.zeros_like(self.data)
        #print("masked data shape: {}".format(self.masked_data.shape))
        projected_coefficients = basis_decomp.projection(self.masked_data)
        #print("projected coefficients size: {}".format(projected_coefficients.size))
        fit_output['projected_coefficients'][...] = projected_coefficients
        projected_data[...] = basis_decomp.image_from_coefficients(projected_coefficients)
        fit_output['projected_image'] = Image2D(projected_data, self.x, self.y, self.z)
        fit_output['projected_image'].apply_mask(self.mask)

    def _optimize_projection(self, basis_decomp, fit_output):
        """
        find the optimized coefficients for a projection of the masked data

        Parameters
        ----------
        basis_decomp: BasisDecomposition2D
            ImageStack of basis functions
        fit_output: dict
            dict for holding fit information
        """
        fitted_data = np.zeros_like(self.data)
        res = basis_decomp._fit_coefficients(self.masked_data,
                                            fit_output['projected_coefficients'])
        fit_output['fitted_coefficients'][...] = res[0]
        cost = np.sum(res[2]['fvec']**2)
        if res[1] is not None:
            diag_vals = np.squeeze(np.array([np.diag(res[1])]))
            diag_vals[diag_vals<0.] = 0.
            diag_vals *= cost/(len(res[2]['fvec'])-res[0].size)
            fit_output['std_devs'][...] = np.sqrt(diag_vals)
        fitted_data[...] = basis_decomp.image_from_coefficients(res[0])
        fit_output['squared_residuals'] = (res[2]['fvec']**2).sum()
        fit_output['fitted_image'] = Image2D(fitted_data, self.x, self.y, self.z)
        fit_output['fitted_image'].apply_mask(self.mask)

    def get_basis_decomp(self, basis, modes):
        if self.basis_decomp is None:
            self.basis_decomp = BasisDecomposition2D(basis, modes, self.x, self.y, self.mask)
        else:
            if not basis == self.basis_decomp.basis:
                self.basis_decomp = None
                return self.get_basis_decomp(basis, modes)
            #first_mode = self.basis_decomp.slice_z(z_value=self.basis_decomp.z[0])
            #if not first_mode._check_image_compatible(self):
            #    self.basis_decomp = None
            #    return self.get_basis_decomp(basis, modes)
            self.basis_decomp.update_modes(modes)
        return self.basis_decomp


    def fit_basis(self, basis, modes):
        """
        fit modes of a basis to the masked image

        Parameters
        ----------
        basis: BasisFunctions Enum
            the basis to use for fitting
        modes: np.ndarray<N,>(int)
            the modes of the basis functions to use
        """
        is_polar = basis_functions.is_polar(basis)
        #mode_start = basis_functions.mode_start(basis)
        #modes = np.arange(mode_start, mode_start+n_modes, dtype=np.int64)
        bd = self.get_basis_decomp(basis, modes)
        fit_output = Image2D.empty_fit_output(modes)
        projected_coefficients = self._basis_projection(bd, fit_output)
        self._optimize_projection(bd, fit_output)
        return fit_output

    def fit_gaussian(self):
        """
        fit and return a gaussian model to the data.

        Parameters
        ----------

        """
        R, PHI = self.get_cyl_dimensions()
        plot = False
        gauss_model = CenteredGaussianModel(plot)
        sort_index = np.argsort(R.flatten())
        r_values = R.flatten()[sort_index]
        data = self.masked_data.flatten()[sort_index]

        gauss_model.fit_parameters(R.flatten(), self.masked_data.flatten())
        model_values = gauss_model.evaluate(R.flatten()).reshape(R.shape)
        #self.data -= model_values
        #print(gauss_model.parameters)
        model_image = Image2D(model_values, self.x, self.y, self.z)
        model_image.apply_mask(self.mask)
        return model_image

    def remove_gaussian(self):
        """
        fit and remove a gaussian from the data.

        Parameters
        ----------

        """
        gaussian_model = self.fit_gaussian()
        self.data = self.data-gaussian_model.data

    def average_edge_data(self):
        complement_mask = Mask2D.from_mask(self.mask, complement=True)
        complement_mask.constraint = complement_mask.constraint*0.95
        XY = self.get_cart_dimensions()
        complement_mask_array = complement_mask.generate_mask(XY)
        complement_masked_data = ma.array(self.data, copy=False)
        complement_masked_data.mask = np.logical_or(complement_mask_array, self.mask.current)
        return complement_masked_data.mean()

    def _check_image_compatible(self, other):
        """
        checks if two images can be compared pixelwise
        """

        if not isinstance(self, type(other)):
            raise ValueError("cannot compare image of type {}".format(type(self)) +
                             " to image of type {}".format(type(other)))

        self_X, self_Y = self.get_cart_dimensions()
        other_X, other_Y = other.get_cart_dimensions()

        self_masked_X = ma.array(self_X, mask=self.mask.current)
        self_masked_Y = ma.array(self_Y, mask=self.mask.current)

        other_masked_X = ma.array(other_X, mask=other.mask.current)
        other_masked_Y = ma.array(other_Y, mask=other.mask.current)

        if not self_masked_X.count() == other_masked_X.count():
            raise ValueError("masked x positions are of different size " +
                             "image1 x size: {} ".format(self_masked_X.count()) +
                             "image2 x size: {}".format(other_masked_X.count()))

        if not self_masked_Y.count() == other_masked_Y.count():
            raise ValueError("masked y positions are of different size " +
                             "image1 y size: {} ".format(self_masked_Y.count()) +
                             "image2 y size: {}".format(other_masked_Y.count()))

        if not np.all(np.isclose(self_masked_X.compressed(), other_masked_X.compressed())):
            raise ValueError("cannot compare images with different unmasked x positions")

        if not np.all(np.isclose(self_masked_Y.compressed(), other_masked_Y.compressed())):
            raise ValueError("cannot compare images with different unmasked y positions")

    def rotate(self, angle):
        rotated_data = ndimage.rotate(self.data, angle, reshape=False)
        self.data = rotated_data

class ImageStack2D(ImageStackBase):

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
            The position of the image in the z direction
        """
        super().__init__(data)
        self.x = x
        self.y = y
        self.z = z
        #self.n_layers = z.size
        self.basis_decomp = None

    @property
    def n_layers(self):
        return self.z.size

    @classmethod
    def from_image_list(image_stack2D, images):
        """
        returns an ImageStack2D of by concatenating a list of images into one
        stack

        Parameters
        ----------
        images: sequence of Image2D objects
            the images to concatenate
        """
        x = images[0].x
        y = images[0].y
        z_vals = np.zeros(len(images))
        data = np.zeros((x.size, y.size, z_vals.size))
        for iz in range(len(images)):
            image = images[iz]
            z_vals[iz] = image.z
            data[:, :, iz] = image.data
        image_stack2D = ImageStack2D(data, x, y, z_vals)
        image_stack2D.apply_mask(images[0].mask)
        return image_stack2D

    @classmethod
    def from_basis(image_stack2D, basis, modes, x, y):
        """
        returns an ImageStack2D of a basis function of evaluated on x/y.

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
        y: nd.array<float>(M,)
            the positions over which to evaluate the basis functions
        """
        polar = basis_functions.is_polar(basis)
        #n_modes = n_max-n_min
        n_modes = modes.size
        #n_modes = modes.size
        data = np.zeros((x.size, y.size, n_modes))
        X, Y = np.meshgrid(x, y, indexing='ij')
        if polar:
            R = np.sqrt(X**2 + Y**2)
            PHI = np.arctan2(Y,X)
            dimension1 = R
            dimension2 = PHI
        else:
            dimension1 = X
            dimension2 = Y
        #modes = np.array(list(range(n_min, n_max+1)), dtype=np.int64)
        basis_func = basis_functions.function(basis)
        for ii, mode in enumerate(modes):
            data[..., ii] = basis_func(mode, dimension1, dimension2)
        return image_stack2D(data, x, y, modes.astype(np.double))

    def slice_z(self, z_index=None, z_value=None):
        """
        slice the current image stack in z direction

        Parameters
        ----------
        z_index: int or sequence of ints
            the indexes to slice at
        z_value: float or sequence of floats
            the z positions to slice at

        Returns
        -------
        ImageStack2D
            if z_index or z_value are sequence
        Image2D
            if z_index or z_value are scalar
        """
        sliced_data, sliced_z, is_sequence = self._slice_z(z_index=z_index, z_value=z_value)
        if is_sequence:
            sliced_image = ImageStack2D(sliced_data, self.x, self.y, sliced_z)
        else:
            sliced_image = Image2D(sliced_data, self.x, self.y, sliced_z)
        sliced_image.apply_mask(mask=self.mask)
        sliced_image.label = self.label
        return sliced_image

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

    def remove_gaussian(self):
        """
        fit and remove a gaussian from the data.

        Parameters
        ----------

        """
        images = []
        for layer in range(self.n_layers):
            sliced_image = self.slice_z(z_index=layer)
            sliced_image.remove_gaussian()
            images.append(sliced_image)
        return ImageStack2D.from_image_list(images)






    @classmethod
    def from_file(image_stack2D, fpath):
        with np.load(fpath, allow_pickle=True) as file_data:
            #file_data = np.load(fpath, allow_pickle=True)
            required_fields = ['data', 'x', 'y', 'z']
            for field in required_fields:
                if field not in file_data:
                    raise KeyError("required field : {}".format(field) +
                                   " is missing from npz file")
            img_stack = image_stack2D(file_data['data'], file_data['x'],
                                      file_data['y'], file_data['z'])
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
            np.savez(fpath, x=self.x, y=self.y, z=self.z,
                     data=self.data, mask_shape=self.mask.shape,
                     mask_region=self.mask.region,
                     mask_constraint=self.mask.constraint)
        else:
            np.savez(fpath, x=self.x, y=self.y, z=self.z,
                     data=self.data)


    def central_value(self):
        x_length = self.x.size
        x_mid = int((x_length-1)/2)
        y_length = self.y.size
        y_mid = int((y_length-1)/2)
        return self.masked_data[x_mid, y_mid, :]

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


    # def flux(self, xy_scale=1.0):
    #     flux = np.zeros_like(self.z)
    #     for i_z in range(self.z.size):
    #         flux[i_z] = np.trapz(self.masked_data[:, i_z],
    #                              self.x*xy_scale)
    #     return flux


    def average_over_dimension(self, dimension='y', use_masked=False):
        """
        average data values over a lateral dimension and return Image1D

        Parameters
        ----------
        dimension: string |'x'|'y'|
            the dimension to average over
        use_masked: bool
            average the masked data
        """
        images = []
        for layer in range(self.n_layers):
            sliced_image = self.slice_z(z_index=layer)
            im_1d = sliced_image.average_over_dimension(dimension=dimension,
                                                        use_masked=use_masked)
            images.append(im_1d)
        print(len(images))
        return ImageStack1D.from_image_list(images)

    def slice_dimension(self, dimension='y', position=0., use_masked=False):
        """
        average data values over a lateral dimension and return Image1D

        Parameters
        ----------
        dimension: string |'x'|'y'|
            the dimension to average over
        position: float
            the position at which to slice the dimension
        use_masked: bool
            average the masked data
        """
        images = []
        for layer in range(self.n_layers):
            sliced_image = self.slice_z(z_index=layer)
            im_1d = sliced_image.slice_dimension(dimension=dimension,
                                                        position=position,
                                                        use_masked=use_masked)
            images.append(im_1d)
        return ImageStack1D.from_image_list(images)

    def rotate(self, angle):
        new_data = np.zeros_like(self.data)
        for layer in range(self.n_layers):
            sliced_image = self.slice_z(z_index=layer)
            rotated_data = ndimage.rotate(sliced_image.data, angle, reshape=False)
            new_data[:, :, layer] = rotated_data
        self.data = new_data

class BasisDecomposition2D(ImageStack2D):
    """
    Fit a set of basis functions to 2D image data
    """

    def __init__(self, basis_func, modes,
                 x, y, mask):
        """
        Parameters
        ----------
        basis_func: BasisFunctions Enum
            The set of functions to use for the decomposition
        modes: nd.array<int>(M,)
            the indices of the modes to use in the decomposition
        x: nd.array<float>(N,)
            the positions over which to evaluate the basis functions
        y: nd.array<float>(P,)
            the positions over which to evaluate the basis functions
        mask: imagestack.mask.Mask object
            mask for the data
        """
        self.basis = basis_func
        self.modes = modes
        self.modes_mask = ma.asarray(self.modes)
        mask = Mask2D.from_mask(mask)
        x, y = BasisDecomposition2D.normalise_dimensions(x, y, mask)
        data = self.init_basis(x, y)
        self.reset_projection_cache()
        if not isinstance(modes, np.ndarray):
            raise AttributeError("modes must by numpy.ndarray")
        super().__init__(data, x, y, modes.astype(float))
        self.apply_mask(mask=mask)

    @staticmethod
    def normalise_dimensions(x, y, mask):
        if mask.polar:
            normed_x =  x/mask.constraint
            normed_y =  y/mask.constraint
            mask.constraint = 1.0
        else:
            normed_x =  x/mask.constraint[0]
            normed_y =  y/mask.constraint[1]
            mask.constraint = np.array([1.0, 1.0])
        return normed_x, normed_y

    def init_basis(self, x, y):
        return ImageStack2D.from_basis(self.basis, self.modes, x, y).data

    def update_modes(self, new_modes):
        #print("updating modes from {} to {}".format(self.modes.size, new_modes.size))
        #if (new_modes.size == self.modes.size and
        #    np.all(np.isclose(new_modes, self.modes))):
        #    print("modes same size")
        #    return

        to_evaluate = []
        for mode in new_modes:
            if not np.any(np.isclose(mode, self.modes)):
                to_evaluate.append(mode)
        if len(to_evaluate) == 0:
            condition = [x not in new_modes for x in self.modes]
            self.modes_mask = ma.masked_where( condition, self.modes)
            #print("all new modes not masked")
            condition = [x not in new_modes for x in self.modes]
            self.modes_mask = ma.masked_where( condition, self.modes)
            self.z = self.modes_mask.compressed()
            #print("modes size: {}".format(self.modes.size))
            return

        to_evaluate = np.array(to_evaluate)
        new_data = ImageStack2D.from_basis(self.basis, to_evaluate, self.x, self.y).data

        total_data = np.concatenate([self.data, new_data], axis=2)
        total_modes = np.concatenate([self.modes, to_evaluate])
        sort_indices = np.argsort(total_modes)
        total_modes = total_modes[sort_indices]
        total_data = total_data[:, :, sort_indices]
        self.modes = total_modes
        self.data = total_data
        condition = [x not in new_modes for x in self.modes]
        self.modes_mask = ma.masked_where( condition, self.modes)
        self.z = self.modes_mask.compressed()
        #print("modes size: {}".format(self.modes.size))

    def reset_projection_cache(self):
        self.projection_cache = {}

    def projection(self, masked_2D_data):
        n_unmasked = ma.count(masked_2D_data)
        #print("n layers: {}".format(self.n_layers))
        #print("masked vals: {}".format(self.modes_mask.count()))
        coefficients = np.zeros(self.n_layers)
        i_coef = 0
        for ilayer, mode in enumerate(self.modes):
            if ma.is_masked(self.modes_mask[ilayer]):
                continue
            if mode in self.projection_cache:
                coef = self.projection_cache[mode]
            else:
                basis_func = self.masked_data[..., ilayer]
                coef = ma.sum(masked_2D_data*basis_func)/n_unmasked
                self.projection_cache[mode] = coef
            coefficients[i_coef] = coef
            i_coef += 1
        #print("coefs size: {}".format(coefficients.size))
        return coefficients

    def image_from_coefficients(self, coefficients):
        all_coefs = np.zeros(self.modes.size)
        all_coefs[~self.modes_mask.mask] = coefficients
        stacked_image_data = (all_coefs*self.masked_data)
        image_data = stacked_image_data.sum(axis=2)
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
        #print("error function output: {}".format(np.amax(diff.ravel())))
        return diff.ravel()

    def err_fun_jac(self, x):
        sliced_data = self.masked_data[:, :, ~self.modes_mask.mask]
        X = self.get_cart_dimensions()[0]
        #print("jacobian output: {}".format(np.amax(sliced_data.flatten())) )
        return sliced_data.reshape(X.size, self.modes_mask.count())

    #def _check_image_compatible(self, other):
    #    print(super())
    #    ImageStack2D._check_image_compatible(super(), other)
