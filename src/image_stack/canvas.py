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
        print(exp)
    try:
        plot_image1D(ax, image, x_scale=length_scale)
        return
    except ValueError as exp:
        print(exp)

    #raise ValueError("unknown image type: {}".format(type(image)))

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


def plot_image_through_focus(ax, image, x_scale=1.0, z_scale=1.0):
    x = image.x*x_scale
    z = image.z*z_scale
    plt.sca(ax)
    vmin = None
    vmax = None
    X, Z = np.meshgrid(x, z, indexing='ij')
    plt.pcolormesh(X, Z, image.masked_data, shading='gouraud',
                    vmin=vmin, vmax=vmax)



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

"""

def combined_plot(XX ,YY, data1, data2, vmin=None, vmax=None,
                  label1="Experiment", label2="Simulation"):

    #Plot two different images with diagnostics

    plot_X1 = XX[0]*1e6
    plot_Y1 = YY[0]*1e6

    plot_X2 = XX[1]*1e6
    plot_Y2 = YY[1]*1e6

    fig = plt.figure(figsize=(13,7))
    gs = gridspec.GridSpec(2, 3, right =0.9, wspace = 0.7, hspace = 0.5)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    box_lim = 0.2

    #abs_dif = np.abs(data1-data2)
    dif = data1-data2
    pp = 2
    plt.sca(ax0)
    plt.pcolormesh(plot_X1[::pp,::pp], plot_Y1[::pp,::pp], data1[::pp,::pp], shading='gouraud', vmin=vmin, vmax=vmax)
    plt.axvline(box_lim , c='w')
    plt.axvline(-box_lim , c='w')
    plt.axhline(box_lim , c='w')
    plt.axhline(-box_lim , c='w')
    plt.title(label1)
    #plt.title("Target")
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    plt.colorbar()

    plt.sca(ax1)
    plt.pcolormesh(plot_X2[::pp,::pp], plot_Y2[::pp,::pp], data2[::pp,::pp], shading='gouraud', vmin=vmin, vmax=vmax)
    plt.axvline(box_lim , c='w')
    plt.axvline(-box_lim , c='w')
    plt.axhline(box_lim , c='w')
    plt.axhline(-box_lim , c='w')
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    plt.title(label2)
    #plt.title("Optimum")
    #plt.imshow(sim)
    plt.colorbar()

    plt.sca(ax2)
    plt.pcolormesh(plot_X1[::pp,::pp], plot_Y1[::pp,::pp], dif[::pp,::pp], shading='gouraud')
    plt.axvline(box_lim , c='w')
    plt.axvline(-box_lim , c='w')
    plt.axhline(box_lim , c='w')
    plt.axhline(-box_lim , c='w')
    plt.title("Difference")
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    #plt.imshow(dif)
    plt.colorbar()

    mid_x = int((plot_X1.shape[0]-1)/2)
    mid_y = int((plot_Y1.shape[1]-1)/2)
    x = plot_X1[:, mid_y]
    y = plot_Y1[mid_x, :]
    data1_x = data1[:, mid_y]
    data1_y = data1[mid_x, :]
    data2_x = data2[:, mid_y]
    data2_y = data2[mid_x, :]
    dif_x = np.abs(data1_x-data2_x)
    dif_y = np.abs(data1_y-data2_y)

    plt.sca(ax3)
    #label1='Tar.'
    #label2='Opt.'
    label1=label1[:4] +'.'
    label2=label2[:4] +'.'
    plt.plot(x, data1_x, label=label1)
    plt.plot(x, data2_x, label=label2)
    plt.ylim([vmin, vmax])
    plt.title("y=0")
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Intensity")

    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 1.0], loc='center')
    plt.sca(ax4)
    plt.plot(y, data1_y, label=label1)
    plt.plot(y, data2_y, label=label2)
    plt.ylim([vmin, vmax])
    plt.title("x=0")
    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 1.0], loc='center')
    plt.xlabel("Y ($\mu$m)")
    plt.ylabel("Intensity")

    plt.sca(ax5)
    plt.plot(x, dif_x, label='y=0')
    plt.plot(y, dif_y, label='x=0')
    plt.title("Difference x/y")
    plt.xlabel("X/Y ($\mu$m)")
    plt.ylabel("Dif. Intensity")
    #plt.ylim([0.0, 0.02])
    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 0.5], loc='center left')
    return fig


def triple_plot(X ,Y, data1, data2, data3, vmin=None, vmax=None,
                  label1="Experiment", label2="Full Refl.", label3="Scat. Only"):

    #Plot three different images with diagnostics
    plot_X = X*1e6
    plot_Y = Y*1e6

    fig = plt.figure(figsize=(13,7))
    gs = gridspec.GridSpec(2, 3, right =0.9, wspace = 0.7, hspace = 0.5)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    box_lim = 0.2
    view_lim = 1.0
    #abs_dif = np.abs(data1-data2)
    #dif = data1-data2
    pp = 2
    plt.sca(ax0)
    plt.pcolormesh(plot_X[::pp,::pp], plot_Y[::pp,::pp], data1[::pp,::pp], shading='gouraud', vmin=vmin, vmax=vmax)
    #plt.axvline(box_lim , c='w')
    #plt.axvline(-box_lim , c='w')
    #plt.axhline(box_lim , c='w')
    #plt.axhline(-box_lim , c='w')
    plt.title(label1)
    plt.xlim((-view_lim, view_lim))
    plt.ylim((-view_lim, view_lim))
    #plt.title("Target")
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    plt.colorbar()

    plt.sca(ax1)
    plt.pcolormesh(plot_X[::pp,::pp], plot_Y[::pp,::pp], data2[::pp,::pp], shading='gouraud', vmin=vmin, vmax=vmax)
    #plt.axvline(box_lim , c='w')
    #plt.axvline(-box_lim , c='w')
    #plt.axhline(box_lim , c='w')
    #plt.axhline(-box_lim , c='w')
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    plt.title(label2)
    plt.xlim((-view_lim, view_lim))
    plt.ylim((-view_lim, view_lim))
    #plt.title("Optimum")
    #plt.imshow(sim)
    plt.colorbar()

    plt.sca(ax2)
    plt.pcolormesh(plot_X[::pp,::pp], plot_Y[::pp,::pp], data3[::pp,::pp], shading='gouraud', vmin=vmin, vmax=vmax)
    #plt.axvline(box_lim , c='w')
    #plt.axvline(-box_lim , c='w')
    #plt.axhline(box_lim , c='w')
    #plt.axhline(-box_lim , c='w')
    plt.title(label3)
    plt.xlim((-view_lim, view_lim))
    plt.ylim((-view_lim, view_lim))
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Y ($\mu$m)")
    #plt.imshow(dif)
    plt.colorbar()

    mid_x = int((plot_X.shape[0]-1)/2)
    mid_y = int((plot_Y.shape[1]-1)/2)
    x = plot_X[:, mid_y]
    y = plot_Y[mid_x, :]
    data1_x = data1[:, mid_y]
    data1_y = data1[mid_x, :]

    data2_x = data2[:, mid_y]
    data2_y = data2[mid_x, :]

    data3_x = data3[:, mid_y]
    data3_y = data3[mid_x, :]

    dif2_x = np.abs(data1_x-data2_x)
    dif2_y = np.abs(data1_y-data2_y)

    dif3_x = np.abs(data1_x-data3_x)
    dif3_y = np.abs(data1_y-data3_y)

    plt.sca(ax3)
    #label1='Tar.'
    #label2='Opt.'
    label1=label1[:4] +'.'
    label2=label2[:4] +'.'
    label3=label3[:4] +'.'
    plt.semilogy(x, data1_x, label=label1)
    plt.semilogy(x, data2_x, label=label2)
    plt.semilogy(x, data3_x, label=label3)
    plt.ylim([vmin, vmax])
    plt.title("y=0")
    plt.xlabel("X ($\mu$m)")
    plt.ylabel("Intensity")

    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 1.0], loc='center')
    plt.sca(ax4)
    plt.semilogy(y, data1_y, label=label1)
    plt.semilogy(y, data2_y, label=label2)
    plt.semilogy(y, data3_y, label=label3)
    plt.ylim([vmin, vmax])
    plt.title("x=0")
    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 1.0], loc='center')
    plt.xlabel("Y ($\mu$m)")
    plt.ylabel("Intensity")

    plt.sca(ax5)
    plt.semilogy(x, dif2_x, label='y=0 '+label2)
    plt.semilogy(y, dif2_y, label='x=0 '+label2)

    plt.semilogy(x, dif3_x, label='y=0 '+label3)
    plt.semilogy(y, dif3_y, label='x=0 '+label3)
    plt.title("Difference to Exp.")
    plt.xlabel("X/Y ($\mu$m)")
    plt.ylabel("Dif. Intensity")
    #plt.ylim([0.0, 0.02])
    plt.legend(fontsize=18., bbox_to_anchor=[1.0, 0.5], loc='center left')
    return fig


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
"""
