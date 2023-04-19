import numpy as np
import numpy.ma as ma
from scipy.special import eval_legendre, eval_chebyt
from enum import Enum

class BasisFunctions(Enum):
    LEGENDRE1D = 0
    LEGENDRE1DX = 1
    LEGENDRE1DY = 2
    ZERNIKE = 3
    ZERNIKE_FRINGE = 3
    ZERNIKE_OSA = 4
    ZERNIKE_FRINGE_CONSTRAINED = 5
    CHEBYSHEV = 6

def function(basis_function):
    if basis_function == BasisFunctions.LEGENDRE1D:
        return legendre1D
    elif basis_function == BasisFunctions.LEGENDRE1DX:
        return legendre1DX
    elif basis_function == BasisFunctions.LEGENDRE1DY:
        return legendre1DY
    elif basis_function == BasisFunctions.ZERNIKE:
        return zernike_fringe
    elif basis_function == BasisFunctions.ZERNIKE_OSA:
        return zernike_osa
    elif basis_function == BasisFunctions.ZERNIKE_FRINGE_CONSTRAINED:
        return zernike_fringe_constrained
    elif basis_function == BasisFunctions.CHEBYSHEV:
        return chebyshev
    else:
        raise ValueError("unknown basis function")

def is_polar(basis_function):
    if basis_function == BasisFunctions.LEGENDRE1D:
        return False
    elif basis_function == BasisFunctions.LEGENDRE1DX:
        return False
    elif basis_function == BasisFunctions.LEGENDRE1DY:
        return False
    elif basis_function == BasisFunctions.ZERNIKE:
        return True
    elif basis_function == BasisFunctions.ZERNIKE_OSA:
        return True
    elif basis_function == BasisFunctions.ZERNIKE_FRINGE_CONSTRAINED:
        return True
    elif basis_function == BasisFunctions.CHEBYSHEV:
        return False
    else:
        raise ValueError("unknown basis function")

def mode_start(basis_function):
    if basis_function == BasisFunctions.LEGENDRE1D:
        return 0
    elif basis_function == BasisFunctions.LEGENDRE1DX:
        return 0
    elif basis_function == BasisFunctions.LEGENDRE1DY:
        return 0
    elif basis_function == BasisFunctions.ZERNIKE:
        return 1
    elif basis_function == BasisFunctions.ZERNIKE_OSA:
        return 1
    elif basis_function == BasisFunctions.ZERNIKE_FRINGE_CONSTRAINED:
        return 1
    elif basis_function == BasisFunctions.CHEBYSHEV:
        return 0
    else:
        raise ValueError("unknown basis function")

def chebyshev(n, x):
    """
    Evaluates chebyshev polynomial of order n in 1 dimension

    Parameters
    ----------
    n: int
        order of polynomial
    x: np.ndarray<float>(N,)
        1d array of x positions
    """
    values = eval_chebyt(n, x)
    return values

def legendre1D(n, x):
    """
    Evaluates legendre polynomial of order n in 1 dimension

    Parameters
    ----------
    n: int
        order of polynomial
    x: np.ndarray<float>(N,)
        1d array of x positions
    """
    leg = eval_legendre(n, x)
    return leg

def legendre1DX(n, x, y):
    """
    Evaluates legendre polynomial of order n in 1 dimension

    Parameters
    ----------
    n: int
        order of polynomial
    x: np.ndarray<float>(N,)
        1d array of x positions
    """
    leg = eval_legendre(n, x)
    return leg

def legendre1DY(n, x, y):
    """
    Evaluates legendre polynomial of order n in 1 dimension

    Parameters
    ----------
    n: int
        order of polynomial
    x: np.ndarray<float>(N,)
        1d array of x positions
    """
    leg = eval_legendre(n, y)
    return leg

def legendre2D(n, x, y):
    """
    Evaluates legendre polynomial of order n in 1 dimension

    Parameters
    ----------
    n: int
        order of polynomial
    x: np.ndarray<float>(N,)
        1d array of x positions
    """

    if n % 2:
        leg = eval_legendre(int(n*0.5)+1, x)
    else:
        leg = eval_legendre(int(n*0.5), y)

    return leg

def is_even(x):
    """check if value is even"""
    return x % 2 != 0

def get_j_from_n_m(n, m):
    '''
    Convert a classical radial, azimuthal (n, m) Zernike polynomial index to a
    Fringe 1D index.
    '''
    j = (1+ (n + np.abs(m))/2)**2 - 2*np.abs(m) + (1-np.sign(m))/2
    return int(j)

def get_n_m_from_j_fringe(j):
    '''
    Convert a Fringe 1D index to the corresponding
    classical radial, azimuthal (n, m) pair.
    '''
    assert j > 0, 'j must be > 0! (Piston is j=1)'

    d = np.floor(np.sqrt(j-1)) + 1
    if not is_even(d**2 - j):
        m = np.ceil( (d**2 - j) / 2. )
    else:
        m = np.ceil( (-d**2 + j -1) / 2. )
    n = 2 * (d - 1) - abs(m)
    return int(n), int(m)

def get_n_m_from_j_osa(j):
    '''
    Convert a Fringe 1D index to the corresponding
    classical radial, azimuthal (n, m) pair.
    '''
    assert j > 0, 'j must be > 0! (Piston is j=1)'
    j -= 1
    n = int(np.ceil((-3.+np.sqrt(9.+8.*j))/2.))
    m = 2*j - n*(n+2)
    return int(n), int(m)

def manhatten_dist(i, j):
    return np.abs(i) + np.abs(j)

def chebyshev_dist(i, j):
    return np.max((np.abs(i), np.abs(j)))

def get_upper_part(side_length):
    vals = np.arange(1,side_length*2,2)
    part = np.tile(vals, side_length).reshape(side_length,side_length).T
    return np.triu(part)

def get_lower_part(side_length):
    part = np.tile(np.arange(2,side_length*2+1,2),side_length).reshape(side_length,side_length)
    return np.tril(part, k=-1)

def get_square(side_length):
    array = np.zeros((side_length, side_length))
    upper = get_upper_part(side_length)
    lower = get_lower_part(side_length)

    for ii in range(side_length):
        for jj in range(side_length):
            #m_dist = manhatten_dist(ii, jj)
            c_dist = chebyshev_dist(ii, jj)

            array[ii, jj] = c_dist**2

    array += upper
    array += lower
    return array

def get_diag_mask(side_length, max_m):
    x = np.arange(side_length)
    y = np.arange(side_length)
    X,Y =np.meshgrid(x, y)
    return np.abs(X-Y)>=max_m

def convert_matrix_pos_to_n_m(i,j):
    m = j-i
    n = np.abs(i) + np.abs(j)
    return n, m

def get_n_m_from_j_fringe_constrained(goal_j, max_m=4):
    max_j = 0
    side_length = 1
    while max_j <= goal_j:
        sq = get_square(side_length)
        mask = get_diag_mask(side_length, max_m)
        valid_positions = ma.masked_where(mask, sq)
        max_j = valid_positions.count()
        side_length += 1

    unique_vals = np.unique(valid_positions)
    val = unique_vals[goal_j-1]
    index = np.where(valid_positions==val)
    n,m = convert_matrix_pos_to_n_m(index[0][0], index[1][0])
    return n, m


def zernike_fringe(j, r, phi):
    n, m = get_n_m_from_j_fringe(j)
    return zernike(n, m, r, phi)

def zernike_osa(j, r, phi):
    n, m = get_n_m_from_j_osa(j)
    return zernike(n, m, r, phi)

def zernike_fringe_constrained(j, r, phi, max_m=4):
    n, m = get_n_m_from_j_fringe_constrained(j, max_m=max_m)
    return zernike(n, m, r, phi)

def zernike_radial_func(n, m, r):
    """
    Fucntion to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (ndarray): 2-d array of radii from the centre the array

    Returns:
        ndarray: The Zernike radial function
    """

    rad_func = np.zeros(r.shape)

    for i in range(0, (n - m) // 2 + 1):

        #check if R is zero for odd n-mreturn projected_stack, coefficients
        rad_func += np.array(r**(n - 2 * i) * ( ((-1)**(i)) *
                         np.math.factorial(n - i) ) /
                         (np.math.factorial(i) *
                          np.math.factorial((n + m)//2 - i) *
                          np.math.factorial((n - m)//2 - i)),
                         dtype='float')
    rad_func[r>1]=0
    return rad_func

def zernike(n, m, r, phi):
    """
    Creates the Zernike polynomial with fringe index j.

    Parameters
    ----------
    n: int
        radial index
    m: int
        azimuthal index
    r: <np.double> np.ndarray
        The radial position
    phi: <np.double> np.ndarray
        The azimuthal angle in radians
    """
    if m==0:
        Z = np.sqrt((n+1)/np.pi)*zernike_radial_func(n, 0, r)
    else:
        if m > 0: # j is even
            Z = np.sqrt(2*(n+1)/np.pi) * zernike_radial_func(n, m, r) * np.cos(m * phi)
        else:   #i is odd
            m = abs(m)
            Z = np.sqrt(2*(n+1)/np.pi) * zernike_radial_func(n, m, r) * np.sin(m * phi)

    return Z
