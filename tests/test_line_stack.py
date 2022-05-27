import pytest
import numpy as np
import os, sys
import logging
sys.path.append(os.path.join("..","src"))
from image_stack.image1d import Image1D, ImageStack1D, BasisDecomposition1D
from image_stack.basis_functions import BasisFunctions
from image_stack.statistics import InformationCriteria
from image_stack.mask import Mask1D
def test_init():
    x = np.linspace(-1., 1.,51)
    data = np.vstack([x**2, x**3]).T
    z = np.array([0., 1.])
    line_stack = ImageStack1D(data, x, z)

@pytest.fixture
def create_line_stack():
    x = np.linspace(-1., 1.,51)
    data = np.vstack([x**2, x**3]).T
    z = np.array([0., 1.])
    line_stack = ImageStack1D(data, x, z)
    yield line_stack

@pytest.fixture
def create_line():
    x = np.linspace(-1., 1.,51)
    data = x**2
    line = Image1D(data, x, 0.)
    yield line

@pytest.fixture
def create_basis_functions():
    x = np.linspace(-1., 1.,51)
    basis = ImageStack1D.from_basis(BasisFunctions.LEGENDRE1D,
                              0, 10, x)
    yield basis


def test_slice_z_index(create_line_stack):
    line_stack = create_line_stack
    line = line_stack.slice_z(z_index=1)

def test_slice_z_value(create_line_stack):
    line_stack = create_line_stack
    line = line_stack.slice_z(z_value=1.)

def test_stack_cart_dimensions(create_line_stack):
    x = create_line_stack.get_cart_dimensions()
    assert(x.size==51)

def test_cyl_dimensions(create_line_stack):
    raises_error = False
    try:
        x = create_line_stack.get_cyl_dimensions()
    except ValueError as excp:
        raises_error = True
    assert(raises_error)

def test_from_basis():
    x = np.linspace(-1., 1.)
    line = ImageStack1D.from_basis(BasisFunctions.LEGENDRE1D, 0, 10, x)

def test_central_val(create_line_stack):
    central_val = create_line_stack.central_value()
    assert(np.isclose(central_val[0], 0.))

def test_flux(create_line_stack):
    flux = create_line_stack.flux()
    assert(np.isclose(flux[0], 0.6672))

def test_add_noise(create_line_stack):
    create_line_stack.add_noise(0.1, seed=1)
    noisy_val = create_line_stack.central_value()
    assert(np.isclose(noisy_val[0], 0.03001703199558275))

def test_apply_mask(create_line_stack):
    line = create_line_stack
    line.set_mask('window', 'edge', 0.2)
    line.apply_mask()
    #print(line.mask.generate_mask(20.0))
    flux = line.flux()
    assert(np.isclose(flux[0], 0.005439999999999997))

def test_transform_data(create_line_stack):
    line = create_line_stack
    line.set_mask('window', 'edge', 0.2)
    line.apply_mask()
    line.transform_data(2, 1.)
    flux = line.flux()
    assert(np.isclose(flux[0], 0.4108799999999999))

def test_normalise_flux(create_line_stack):
    line = create_line_stack
    line.normalise_flux()
    assert(np.isclose(line.flux()[0], 1.0))

def test_normalise_range(create_line_stack):
    line = create_line_stack
    line.transform_data(2, 1.)
    line.normalise_range()
    assert(np.all(np.isclose(line.max(), 1.0)))
    assert(np.all(np.isclose(line.min(), 0.0)))

def test_normalise_highest(create_line_stack):
    line = create_line_stack
    line.transform_data(2, 1.)
    line.normalise_highest()
    assert(np.all(np.isclose(line.max(), 1.0)))

def test_projection(create_basis_functions, create_line):
    coefs = create_basis_functions.projection(create_line)
    assert(np.isclose(coefs[0], 1.76800000e+01))

def test_basis_decomposition():
    x = np.linspace(-15., 15., 51)
    mask = Mask1D('window','edge', 13.)
    mask.generate_mask(x)
    modes = np.array(list(range(10)))
    bd = BasisDecomposition1D(BasisFunctions.LEGENDRE1D,
                              modes, x, mask)

def test_opt_coefs(create_line_stack):
    create_line_stack.set_mask('window', 'edge', 1.0)
    n_modes = create_line_stack.optimise_basis_size(BasisFunctions.LEGENDRE1D,
                                              60, min_n_modes=10)
    print(n_modes)

def test_from_images():
    x = np.linspace(-1., 1.,51)
    data = np.vstack([x**2, x**3]).T
    z = np.array([0., 1.])
    line1 = Image1D(data[:, 0], x, z[0])
    line2 = Image1D(data[:, 1], x, z[1])
    im_stack = ImageStack1D.from_image_list([line1, line2])
