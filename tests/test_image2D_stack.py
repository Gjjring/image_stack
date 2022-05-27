import pytest
import numpy as np
import os, sys
import logging
sys.path.append(os.path.join("..","src"))
from image_stack.image2d import Image2D, ImageStack2D, BasisDecomposition2D
from image_stack.basis_functions import BasisFunctions
from image_stack.statistics import InformationCriteria
from image_stack.mask import Mask2D

def test_init():
    x = np.linspace(-1., 1.,51)
    y = np.linspace(-1., 1., 31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data1 = X**2 + Y**2
    data2 = X**3 + Y**3
    data = np.concatenate([data1[..., None], data2[..., None]], axis=2)
    z = np.array([0., 1.])
    image_stack = ImageStack2D(data, x, y, z)

@pytest.fixture
def create_image_stack():
    x = np.linspace(-1., 1.,51)
    y = np.linspace(-1., 1., 31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data1 = X**2 + Y**2
    data2 = X**3 + Y**3
    data = np.concatenate([data1[..., None], data2[..., None]], axis=2)
    z = np.array([0., 1.])
    image_stack = ImageStack2D(data, x, y, z)
    yield image_stack

@pytest.fixture
def create_unitary_image_stack():
    x = np.linspace(-1., 1., 51)
    y = np.linspace(-1., 1., 51)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data1 = np.ones_like(X)
    data2 = np.ones_like(X)
    data = np.concatenate([data1[..., None], data2[..., None]], axis=2)
    z = np.array([0., 1.])
    image_stack = ImageStack2D(data, x, y, z)
    yield image_stack


@pytest.fixture
def create_image():
    x = np.linspace(-1., 1.,51)
    y = np.linspace(-1., 1.,31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data = X**2 + Y**2
    image = Image2D(data, x, y, 0.)
    yield image

@pytest.fixture
def create_basis_functions():
    x = np.linspace(-1., 1., 51)
    y = np.linspace(-1., 1., 31)
    basis = ImageStack2D.from_basis(BasisFunctions.LEGENDRE2D,
                                    0, 10, x, y)
    yield basis


def test_slice_z_index(create_image_stack):
    image_stack = create_image_stack
    image = image_stack.slice_z(z_index=1)

def test_slice_z_value(create_image_stack):
    image_stack = create_image_stack
    image = image_stack.slice_z(z_value=1.)

def test_stack_cart_dimensions(create_image_stack):
    X, Y = create_image_stack.get_cart_dimensions()
    assert(X.shape[0]==51)
    assert(X.shape[1]==31)
    assert(Y.shape[0]==51)
    assert(Y.shape[1]==31)

def test_cyl_dimensions(create_image_stack):
    R, PHI = create_image_stack.get_cyl_dimensions()

def test_from_basis():
    x = np.linspace(-1., 1.)
    y = np.linspace(-1., 1.)
    image = ImageStack2D.from_basis(BasisFunctions.LEGENDRE2D, 0, 10, x, y)

def test_central_val(create_image_stack):
    central_val = create_image_stack.central_value()
    assert(np.isclose(central_val[0], 0.))

def test_flux(create_unitary_image_stack):
    flux = create_unitary_image_stack.flux()
    assert(np.isclose(flux[0], 4.0))

def test_add_noise(create_unitary_image_stack):
    create_unitary_image_stack.add_noise(0.1, seed=1)
    std = create_unitary_image_stack.std()
    assert(np.isclose(std[0], 0.1, rtol=0.1))

def test_apply_mask_rect(create_image_stack):
    image = create_image_stack
    image.set_mask('rectangular', 'edge', np.array([0.2, 0.2]))
    image.apply_mask()
    #print(image.mask.generate_mask(20.0))
    flux = image.flux()
    assert(np.isclose(flux[0], 0.004427851851851848))

def test_apply_mask_circ(create_image_stack):
    image = create_image_stack
    image.set_mask('circular', 'edge', 0.7)
    image.apply_mask()
    #print(image.mask.generate_mask(20.0))
    flux = image.flux()
    assert(np.isclose(flux[0], 0.2379794504181573))

def test_transform_data(create_image_stack):
    image = create_image_stack
    image.set_mask('rectangular', 'edge', np.array([0.2, 0.2]))
    image.apply_mask()
    image.transform_data(2, 1.)
    flux = image.flux()
    assert(np.isclose(flux[0], 0.16885570370370362))

def test_normalise_flux(create_image_stack):
    image = create_image_stack
    image.normalise_flux()
    assert(np.isclose(image.flux()[0], 1.0))

def test_normalise_range(create_image_stack):
    image = create_image_stack
    image.transform_data(2, 1.)
    image.normalise_range()
    assert(np.all(np.isclose(image.max(), 1.0)))
    assert(np.all(np.isclose(image.min(), 0.0)))

def test_normalise_highest(create_image_stack):
    image = create_image_stack
    image.transform_data(2, 1.)
    image.normalise_highest()
    assert(np.all(np.isclose(image.max(), 1.0)))

def test_projection(create_basis_functions, create_image):
    coefs = create_basis_functions.projection(create_image)
    assert(np.isclose(coefs[0], 1110.2133333333336))

def test_average_over_dimension(create_image_stack):
    line = create_image_stack.average_over_dimension()

def test_from_images():
    x = np.linspace(-1., 1.,51)
    y = np.linspace(-1., 1., 31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    z = np.array([0., 1.])
    data1 = X**2 + Y**2
    data2 = X**3 + Y**3
    image1 = Image2D(data1, x, y, z[0])
    image2 = Image2D(data2, x, y, z[1])
    im_stack = ImageStack2D.from_image_list([image1, image2])

def test_basis_decomposition():
    x = np.linspace(-15., 15., 51)
    y = np.linspace(-15., 15., 31)
    mask = Mask2D('rectangular','edge', np.array([13., 13.]))
    mask.generate_mask(x)
    modes = np.array(list(range(10)))
    bd = BasisDecomposition2D(BasisFunctions.LEGENDRE2D,
                              modes, x, y, mask)

def test_opt_coefs(create_image_stack):
    create_image_stack.set_mask('rectangular', 'edge', [1.0, 1.0])
    n_modes = create_image_stack.optimise_basis_size(BasisFunctions.LEGENDRE2D,
                                              60, min_n_modes=10)
    print(n_modes)
