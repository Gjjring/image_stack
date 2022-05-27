import pytest
import numpy as np
import os, sys
import logging
sys.path.append(os.path.join("..","src"))
from image_stack.image2d import Image2D
from image_stack.basis_functions import BasisFunctions
from image_stack.statistics import InformationCriteria

def test_init():
    x = np.linspace(-1., 1., 51)
    y = np.linspace(-1., 1., 31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data = X**2 + Y**2
    z = 0.
    image = Image2D(data, x, y, z)

@pytest.fixture
def create_image():
    x = np.linspace(-1., 1., 51)
    y = np.linspace(-1., 1., 31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data = X**2 + Y**2
    image = Image2D(data, x, y, 0.)
    yield image


def test_cart_dimensions(create_image):
    X, Y = create_image.get_cart_dimensions()
    assert(X.shape[0]==51)
    assert(X.shape[1]==31)
    assert(Y.shape[0]==51)
    assert(Y.shape[1]==31)

def test_cyl_dimensions(create_image):
    R, PHI = create_image.get_cyl_dimensions()

def test_from_basis():
    x = np.linspace(-1., 1.)
    y = np.linspace(-1., 1.)
    image = Image2D.from_basis(BasisFunctions.LEGENDRE2D, 0, x, y)

def test_central_val(create_image):
    central_val = create_image.central_value()
    assert(np.isclose(central_val, 0.))

def test_flux(create_image):
    flux = create_image.flux()
    assert(np.isclose(flux, 2.6706962962962963))

def test_add_noise(create_image):
    create_image.add_noise(0.1, seed=1)
    noisy_val = create_image.central_value()
    assert(np.isclose(noisy_val, -0.20863905654565262))

def test_apply_mask_rect(create_image):
    image = create_image
    image.set_mask('rectangular', 'edge', np.array([0.2, 0.2]))
    image.apply_mask()
    #print(image.mask.generate_mask(20.0))
    flux = image.flux()
    assert(np.isclose(flux, 0.004427851851851848))

def test_apply_mask_circ(create_image):
    image = create_image
    image.set_mask('circular', 'edge', 0.2)
    image.apply_mask()
    #print(image.mask.generate_mask(20.0))
    flux = image.flux()
    assert(np.isclose(flux, 0.01735984405458089))


def test_transform_data(create_image):
    image = create_image
    image.set_mask('rectangular', 'edge', np.array([0.2, 0.2]))
    image.apply_mask()
    image.transform_data(2, 1.)
    flux = image.flux()
    assert(np.isclose(flux, 0.16885570370370362))

def test_normalise_flux(create_image):
    image = create_image
    image.normalise_flux()
    assert(np.isclose(image.flux(), 1.0))

def test_normalise_range(create_image):
    image = create_image
    image.transform_data(2, 1.)
    image.normalise_range()
    assert(np.isclose(image.max(), 1.0))
    assert(np.isclose(image.min(), 0.0))

def test_normalise_highest(create_image):
    image = create_image
    image.transform_data(2, 1.)
    image.normalise_highest()
    assert(np.isclose(image.max(), 1.0))

def test_fit_basis_legendre():
    x = np.linspace(-15., 15.,51)
    y = np.linspace(-15., 15.,31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data = X**2 + Y**2
    image = Image2D(data, x, y, 0.)
    image.set_mask('rectangular','edge', np.array([13., 13.]))
    image.apply_mask()
    fit_output = image.fit_basis(BasisFunctions.LEGENDRE2D, 10)
    for key, val in fit_output.items():
        print(key, val)

def test_fit_basis_zernike():
    x = np.linspace(-15., 15.,51)
    y = np.linspace(-15., 15.,31)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    data = R**2 * np.sin(2*PHI)
    image = Image2D(data, x, y, 0.)
    image.set_mask('circular','edge', 13.)
    image.apply_mask()
    fit_output = image.fit_basis(BasisFunctions.ZERNIKE, 10)
    for key, val in fit_output.items():
        print(key, val)

def test_pixel_comp():
    x1 = np.linspace(-100., 100., 51)
    y1 = np.linspace(-100., 100., 51)
    X1, Y1 = np.meshgrid(x1, y1, indexing='ij')

    R1 = np.sqrt(X1**2 + Y1**2)
    PHI1 = np.arctan2(Y1, X1)
    data1 = R1 * np.sin(PHI1)

    x2 = np.linspace(-200., 200., 101)
    y2 = np.linspace(-200., 200., 101)
    X2, Y2 = np.meshgrid(x2, y2, indexing='ij')

    R2 = np.sqrt(X2**2 + Y2**2)
    PHI2 = np.arctan2(Y2, X2)
    data2 = R2 * np.sin(PHI2) + 1
    
    z = 0.
    image1 = Image2D(data1, x1, y1, z)
    image2 = Image2D(data2, x2, y2, z)
    image2.set_mask('rectangular', 'edge', np.array([100., 100.]))

    pixel_comp = image1.pixel_comparison(image2)
    assert(np.isclose(pixel_comp['squared_dif'],2601.))
