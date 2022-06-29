import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from image_stack.image1d import Image1D, ImageStack1D
from image_stack.image2d import Image2D, ImageStack2D

def plot_image(ax, image, length_scale=1.0):
    try:
        plot_image2D(ax, image, xy_scale=length_scale)
        return
    except ValueError as exp:
        pass
        #print(exp)
    try:
        plot_image1D(ax, image, x_scale=length_scale)
        return
    except ValueError as exp:
        pass
        #print(exp)

    #raise ValueError("unknown image type: {}".format(type(image)))

def plot_image_difference(axes, image1, image2, length_scale=1.0):
    if isinstance(image1, Image2D):
        plot_image_difference_2D(axes, image1, image2, length_scale)
    elif isinstance(image1, Image1D):
        plot_image_difference_1D(axes, image1, image2, length_scale)
    else:
        raise ValueError("unknown image type")

def plot_image_difference_2D(axes, image1, image2, length_scale):
    assert(len(axes)==3)
    images = [image1, image2, image1.abs_dif(image2)]
    for ax, image in zip(axes, images):
        plot_image(ax, image, length_scale=length_scale)
        plt.xlim(np.array(image.xlim())*length_scale)
        plt.ylim(np.array(image.ylim())*length_scale)
        plt.colorbar()

def plot_image_difference_1D(axes, image1, image2, length_scale):
    assert(len(axes)==2)
    plot_axes = [axes[0], axes[0], axes[1]]
    images = [image1, image2, image1.abs_dif(image2)]
    for ax, image in zip(plot_axes, images):
        plot_image(ax, image, length_scale=length_scale)
        plt.xlim(np.array(image.xlim())*length_scale)

def plot_image1D(ax, image, x_scale=1.0):
    x = image.x*x_scale
    plt.sca(ax)
    plt.plot(x, image.masked_data, label=image.label)

def plot_image2D(ax, image, xy_scale=1.0):
    X, Y = image.get_cart_dimensions()
    X *= xy_scale
    Y *= xy_scale
    plt.sca(ax)
    vmin = None
    vmax = None
    plt.pcolormesh(X, Y, image.masked_data, shading='gouraud',
                  vmin=vmin, vmax=vmax)


def plot_image_through_focus(ax, image, x_scale=1.0, z_scale=1.0,
                             vmin=None, vmax=None):
    x = image.x*x_scale
    z = image.z*z_scale
    plt.sca(ax)
    X, Z = np.meshgrid(x, z, indexing='ij')
    plt.pcolormesh(X, Z, image.masked_data, shading='gouraud',
                    vmin=vmin, vmax=vmax)
