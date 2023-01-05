import numpy as np
from imagestack.image1D import ImageStack1D
from imagestack.image2D import ImageStack2D
from imagestack.mask import Mask1D, Mask2D
class BasisDecomposition():

    def __init__(self, image, basis_func, max_n_modes, modes_start,
                 dimensions, mask, polar):
        self.image = image
        self.basis_func = basis_func
        self.max_n_modes = max_n_modes
        self.current_max_n_modes = max_n_modes
        self.modes_start = modes_start
        self.dimensions = dimensions
        self.mask = mask.from_mask(mask)
        self.polar = polar
        self.normalise_dimensions()
        self.init_basis()

    def normalise_dimensions(self):
        if isinstance(mask, Mask1D):
            self.dimensions[0] /= mask.constraint
            mask.constraint = 1.0
        elif isinstance(mask, Mask2D):
            if self.polar:
                self.dimensions[0] /= mask.constraint[0]
                self.dimensions[1] /= mask.constraint[1]
                mask.constraint = [1.0, 1.0]
            else:
                self.dimensions[0] /= mask.constraint
                self.dimensions[1] /= mask.constraint
                mask.constraint = 1.
        else:
            raise ValueError("Unknown mask of type: {}".format(type(mask)))


    def init_basis(self):
        if len(dimensions) == 1:
            self.basis_functions = ImageStack1D.from_basis(self.basis_func,
                                                           self.modes_start,
                                                           self.max_n_modes,
                                                           self.dimensions[0])
        elif len(dimensions) == 2:
            self.basis_functions = ImageStack2D.from_basis(self.basis_func,
                                                           self.max_n_modes,
                                                           self.modes_start,
                                                           self.dimensions[0],
                                                           self.dimensions[1])
        else:
            raise ValueError("dimensions parameter must be length 1 or 2 iterable")
        self.basis_functions.apply_mask(self.mask_shape, self.mask_region, constraint)

    def set_current_max_n_modes(self, n_modes):
        self.current_max_n_modes = n_modes

    def project_coefficients(self):
        return self.basis_functions.projection(self.image)

    def image_from_coefficients(self, coefficients):
        data = self.basis_functions.masked_data[:, :, :self.current_max_n_modes]
        stacked_image_data = (coefficients*data)
        image_data = stacked_image_data.sum(axis=2)
        return image_data

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
