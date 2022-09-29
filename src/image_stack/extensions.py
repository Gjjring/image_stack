from image_stack.image2d import Image2D
from image_stack.image1d import Image1D
import numpy as np
import scipy.constants
#from image_stack.image_base import ImageBase, ImageStackBase

def cartesianfields_to_image2D(cart_field, z, xy_scale=1.0):
    """
    returns a 2D image based on a cartesian field from JCMsuite

    Parameters
    ----------
    cart_field: cartesian field post process from JCMsuite
        the cartesian field holding the x, y and pixel data
    z: float
        the z position to set the image at
    xy_scale: float
        optional scaling for the lateral dimensions
    """
    if not isinstance(cart_field, dict):
        raise ValueError("cart field must be of type dict not :"+
                         " {}".format(type(cart_field)))
    x_shape = cart_field['X'].shape
    x_length = x_shape[0]
    y_length = x_shape[1]
    x_mid = int((x_length-1)/2)
    y_mid = int((y_length-1)/2)
    image_data = np.zeros((x_length, y_length))
    x = np.unique(cart_field['X'])*xy_scale
    y = np.unique(cart_field['Y'])*xy_scale
    n_pol = len(cart_field['field'])
    for i_pol in range(n_pol):
        if cart_field['field'][i_pol].shape[-1] == 3:
            #print("vector field detected")
            image_data += np.linalg.norm(cart_field['field'][i_pol], axis=2)**2
        else:
            #print(intensity.shape)
            #print("scalar field detected")
            update = 4.*np.squeeze(np.real(cart_field['field'][i_pol]))/scipy.constants.epsilon_0
            #print(update.shape)
            image_data += update
        image_data /= n_pol

    #X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    image = Image2D(image_data, x, y, z)
    return image

def cartesianfields_to_image1D(cart_field, z, x_scale=1.0):
    """
    returns a 2D image based on a cartesian field from JCMsuite

    Parameters
    ----------
    cart_field: cartesian field post process from JCMsuite
        the cartesian field holding the x and pixel data
    z: float
        the z position to set the image at
    x_scale: float
        optional scaling for the lateral dimensions
    """
    if not isinstance(cart_field, dict):
        raise ValueError("cart field must be of type dict not :"+
                         " {}".format(type(cart_field)))
    x_shape = cart_field['X'].shape
    x_length = x_shape[0]
    x_mid = int((x_length-1)/2)
    image_data = np.zeros(x_length)
    x = np.unique(cart_field['X'])*x_scale
    n_pol = len(cart_field['field'])
    for i_pol in range(n_pol):
        if cart_field['field'][i_pol].shape[-1] == 3:
            #print("vector field detected")
            image_data += np.linalg.norm(cart_field['field'][i_pol], axis=-1)**2
        else:
            #print(intensity.shape)
            #print("scalar field detected")
            update = 4.*np.squeeze(np.real(cart_field['field'][i_pol]))/scipy.constants.epsilon_0
            #print(update.shape)
            image_data += update
        image_data /= n_pol

    #X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    image = Image1D(image_data, x, z)
    return image
