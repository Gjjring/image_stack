import pytest
import numpy as np
import os, sys
import logging
sys.path.append(os.path.join("..","src"))
from imagestack.image1d import Image1D, BasisDecomposition1D
from imagestack.basis_functions import BasisFunctions
from imagestack.statistics import InformationCriteria
from imagestack.mask import Mask1D

def test_init():
    x = np.linspace(-1., 1.,51)
    data = x**2
    z = 0.
    line = Image1D(data, x, z)

@pytest.fixture
def create_line():
    x = np.linspace(-1., 1.,51)
    data = x**2
    line = Image1D(data, x, 0.)
    yield line


def test_cart_dimensions(create_line):
    x = create_line.get_cart_dimensions()
    assert(x.size==51)

def test_cyl_dimensions(create_line):
    raises_error = False
    try:
        x = create_line.get_cyl_dimensions()
    except ValueError as excp:
        raises_error = True
    assert(raises_error)

def test_from_basis():
    x = np.linspace(-1., 1.)
    line = Image1D.from_basis(BasisFunctions.LEGENDRE1D, 0, x)

def test_central_val(create_line):
    central_val = create_line.central_value()
    assert(np.isclose(central_val, 0.))

def test_flux(create_line):
    flux = create_line.flux()
    assert(np.isclose(flux, 0.6672))

def test_add_noise(create_line):
    create_line.add_noise(0.1, seed=1)
    noisy_val = create_line.central_value()
    assert(np.isclose(noisy_val, -0.06837278591743332))

def test_apply_mask(create_line):
    line = create_line
    line.set_mask('window', 'edge', 0.2)
    line.apply_mask()
    #print(line.mask.generate_mask(20.0))
    flux = line.flux()
    assert(np.isclose(flux, 0.005439999999999997))

def test_transform_data(create_line):
    line = create_line
    x = np.linspace(-15., 15., 51)
    mask = Mask1D('window','edge', 13.)
    mask.generate_mask(x)
    modes = np.array(list(range(10)))
    bd = BasisDecomposition1D(BasisFunctions.LEGENDRE1D,
                              modes, x, mask)

    line.set_mask('window', 'edge', 0.2)
    line.apply_mask()
    line.transform_data(2, 1.)
    flux = line.flux()
    assert(np.isclose(flux, 0.4108799999999999))

def test_normalise_flux(create_line):
    line = create_line
    line.normalise_flux()
    assert(np.isclose(line.flux(), 1.0))

def test_normalise_range(create_line):
    line = create_line
    line.transform_data(2, 1.)
    line.normalise_range()
    assert(np.isclose(line.max(), 1.0))
    assert(np.isclose(line.min(), 0.0))

def test_normalise_highest(create_line):
    line = create_line
    line.transform_data(2, 1.)
    line.normalise_highest()
    assert(np.isclose(line.max(), 1.0))

def test_fit_basis():
    x = np.linspace(-1., 1.,51)
    data = x
    line = Image1D(data, x, 0.)
    line.set_mask('window','edge', 1.)
    line.apply_mask()
    fit_output = line.fit_basis(BasisFunctions.LEGENDRE1D, 10)
    assert(np.isclose(fit_output['fitted_coefficients'][1], 1.0))


def test_opt_coeffs(create_line):
    create_line.set_mask('window', 'edge', 1.0)
    x = np.linspace(-1., 1., 101)
    #data = x# - Y**2
    noise = np.random.normal(0,0.01, x.size)
    data = (46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 -63)/256 + noise
    z = 0.
    image = Image1D(data, x, z)
    image.set_mask('window','edge', 1.)
    image.apply_mask()
    modes_required = image.optimise_basis_size(BasisFunctions.LEGENDRE1D, 50, min_n_modes = 1,
                                               criterion=InformationCriteria.BIC)
    assert(modes_required==11)

def test_pixel_comp():
    x = np.linspace(-100., 100., 51)
    data = 0.5*(3*x**2 - 1) / 100**2#
    data2 = 0.5*(3*x**2 - 1) / 100**2 + 1#
    z = 0.
    image1 = Image1D(data, x, z)
    image1.set_mask('window','edge', 50.)
    image1.apply_mask()
    image2 = Image1D(data2, x, z)
    image2.set_mask('window','edge', 50.)
    image2.apply_mask()
    pixel_comp = image1.pixel_comparison(image2)
    assert(np.isclose(pixel_comp['squared_dif'], 25.))
