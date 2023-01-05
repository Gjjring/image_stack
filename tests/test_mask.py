import pytest
import numpy as np
import os, sys
import logging
sys.path.append(os.path.join("..","src"))
from imagestack.mask import Mask1D, Mask2D

@pytest.fixture
def create_mask1d():
    mask = Mask1D('window', 'edge', 1.)
    yield mask

def test_copy_1d(create_mask1d):
    mask1D = create_mask1d
    mask1D_2 = type(mask1D).from_mask(mask1D)
    mask1D_2.constraint = 5.


@pytest.fixture
def create_mask2d():
    mask = Mask2D('rectangular', 'edge', np.array([1., 1.]))
    yield mask

def test_copy_2d(create_mask2d):
    mask2D = create_mask2d
    mask2D_2 = type(mask2D).from_mask(mask2D)
    mask2D_2.shape = 'circular'
