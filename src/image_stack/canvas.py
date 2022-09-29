import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from image_stack.image1d import Image1D, ImageStack1D
from image_stack.image2d import Image2D, ImageStack2D

def plot_image(ax, image, length_scale=1.0):
    try:
        return plot_image2D(ax, image, xy_scale=length_scale)
    except ValueError as exp:
        pass
        #print(exp)
    try:
        return plot_image1D(ax, image, x_scale=length_scale)
    except ValueError as exp:
        pass
        #print(exp)

    #raise ValueError("unknown image type: {}".format(type(image)))

def plot_image_difference(axes, image1, image2, length_scale=1.0):
    try:
        plot_image_difference_2D(axes, image1, image2, length_scale)
        return
    except ValueError as exp:
        pass
    try:
        plot_image_difference_1D(axes, image1, image2, length_scale)
        return
    except ValueError as exp:
        pass
    raise ValueError("unknown image type")

def plot_image_difference_2D(axes, image1, image2, length_scale,
                             vmin=None, vmax=None, colormaps=None):
    assert(len(axes)==3)
    images = [image1, image2, image1.dif(image2)]
    cbs = []
    for ii, ax, image in zip(range(3), axes, images):
        if ii < 2:
            vmin = image1.min()
            vmax = image1.max()
        else:
            vmin = None
            vmax = None
        if colormaps == None:
            cmap = None
        else:
            cmap = colormaps[ii]

        plot_image2D(ax, image, xy_scale=length_scale, vmin=vmin, vmax=vmax)
        plt.set_cmap(cmap)
        plt.xlim(np.array(image.xlim())*length_scale)
        plt.ylim(np.array(image.ylim())*length_scale)
        cb = plt.colorbar()
        cbs.append(cb)
    return cbs


def plot_image_difference_1D(axes, image1, image2, length_scale):
    assert(len(axes)==2)
    plot_axes = [axes[0], axes[0], axes[1]]
    images = [image1, image2, image1.abs_dif(image2)]
    for ax, image in zip(plot_axes, images):
        plot_image(ax, image, length_scale=length_scale)
        plt.xlim(np.array(image.xlim())*length_scale)

def plot_image1D(ax, image, x_scale=1.0, color=None, ls='-'):
    x = image.x*x_scale
    plt.sca(ax)
    if color is None:
        #color_cycler = plt.rcParams["axes.prop_cycle"]
        #print(type(color_cycler))
        #color = next(color_cycler)
        color = next(ax._get_lines.prop_cycler)['color']
    return plt.plot(x, image.masked_data, label=image.label, color=color, ls=ls)

def plot_image2D(ax, image, xy_scale=1.0, vmin=None, vmax=None):
    X, Y = image.get_cart_dimensions()
    X *= xy_scale
    Y *= xy_scale
    plt.sca(ax)
    return plt.pcolormesh(X, Y, image.masked_data, shading='gouraud',
                          vmin=vmin, vmax=vmax)


def plot_image_through_focus(ax, image, x_scale=1.0, z_scale=1.0,
                             vmin=None, vmax=None):
    x = image.x*x_scale
    z = image.z*z_scale
    plt.sca(ax)
    X, Z = np.meshgrid(x, z, indexing='ij')
    plt.pcolormesh(X, Z, image.masked_data, shading='gouraud',
                    vmin=vmin, vmax=vmax, cmap='Greys_r')

def plot_image2D_stacked(ax, image_stack, xy_scale=1.0, z_scale=1.0,
                         vmin=None, vmax=None):
    for z_index in np.arange(image_stack.z.size):
        #if z_index > 0:
        #    break
        image2D = image_stack.slice_z(z_index=z_index)
        r = image_stack.mask.constraint[0]
        image2D.set_mask('rectangular', 'edge', [r, r])
        X,Y = image2D.get_cart_dimensions()
        X *= xy_scale
        Y *= xy_scale

        X = X[~image2D.mask.current]
        unique_x = np.unique(X)
        x_length = unique_x.size

        Y = Y[~image2D.mask.current]
        unique_y = np.unique(Y)
        y_length = unique_y.size
        plt.sca(ax)

        #Z = np.ones(X.shape)*image2D.z
        Z = image2D.masked_data*1e-3 + image_stack.z[z_index] * z_scale
        Z = Z[~image2D.mask.current].reshape(x_length, y_length)
        X = X.reshape(x_length, y_length)
        Y = Y.reshape(x_length, y_length)
        Z = Z.T
        Z = Z[::-1, ::1]
        ax.plot_surface(X, Y, Z, vmin=vmin, vmax=vmax, cmap='Greys_r',
                        linewidth=0, alpha=1.0)
