import numpy as np

def mask_from_data(data):
    if len(data.shape) == 2:
        return Mask2D(None,None,None)
    elif len(data.shape) == 1:
        return Mask1D(None,None,None)
    else:
        raise ValueError("wrong number of dimensions for mask generation: {}".format(len(data.shape)))

def copy_mask(mask):
    if isinstance(mask, Mask2D):
        return Mask2D.from_mask(mask)
    elif isinstance(mask, Mask1D):
        return Mask1D.from_mask(mask)
    else:
        raise ValueError("wrong object type for mask generation: {}".format(type(mask)))

def get_complement(region):
    """
    returns a region wich is the complement of the input region

    Parameters
    ----------
    region: str
      the region for which to take the complement

    Returns
    -------
    complement: str
      the region which describes the complement of region
    """
    if region == "center":
        return "edge"
    elif region == "edge":
        return "center"
    elif region == "none":
        return "none"


class Mask2D():

    """
    predefined masks for pixel 2D image data

    Attributes
    ----------
    shape: str
        shape of the mask
    region: str
        together with shape defines where the mask is applied
    constraint: float or np.array<float>(2,)
        one or two constraints for the mask
    """


    def __init__(self, shape, region, constraint,
                 origin=np.array([0., 0.])):
        """
        Parameters
        ----------
        shape: str
            shape of the mask`
        region: str
            together with shape defines where the mask is applied
        constraint: float or np.array<float>(2,)
            one or two constraints for the mask
        """
        self.shape = shape
        self.polar = False
        self.region = region
        self.constraint = constraint
        self.current = None
        self.origin = origin


    @classmethod
    def from_mask(Mask2D, other, complement=False):
        if complement:
            region = get_complement(other.region)
        else:
            region = other.region
        new_mask = Mask2D(other.shape, region, np.array(other.constraint), origin=np.array(other.origin))
        new_mask.polar = other.polar
        return new_mask

    def __repr__(self):
        return "Mask2D({},{},{},{})".format(self.shape, self.region, self.constraint, self.origin)

    def generate_mask(self, XY):
        """
        Return a mask array based on shape, region and constraint
        """
        X = XY[0]
        Y = XY[1]
        if self.shape == 'circular':
            mask = self._get_circular_mask(X, Y)
            self.polar = True
        elif self.shape == 'rectangular':
            mask = self._get_rectangular_mask(X, Y)
            self.polar = False
        elif self.shape == None:
            mask = self._get_null_mask(X, Y)
            self.polar = False
        else:
            raise ValueError("unknown mask type for 2D mask :[{}]".format(self.shape))
        self.current = mask
        return mask

    def _get_null_mask(self, X, Y):
        mask = np.full(X.shape, False, dtype=bool)
        return mask

    def _get_circular_mask(self, X, Y):
        R = np.sqrt((X-self.origin[0])**2 + (Y-self.origin[1])**2)
        radius = self.constraint
        if self.region == 'center':
            mask = R<radius
        elif self.region =='edge':
            mask = R>radius
        elif self.region =='none':
            mask = np.full(R.shape, False, dtype=bool)
        else:
            raise ValueError("mask region name {} invalid".format(self.region))
        return mask

    def _get_rectangular_mask(self, X, Y):
        #R = np.sqrt(X**2 + Y**2)
        xy_width = self.constraint
        if self.region == 'center':
            mask = np.abs(X-self.origin[0]) < xy_width[0]
            mask += np.abs(Y-self.origin[1]) < xy_width[1]
        elif self.region =='edge':
            mask = np.abs(X-self.origin[0]) > xy_width[0]
            mask += np.abs(Y-self.origin[1]) > xy_width[1]
        elif self.region =='none':
            mask = np.full(X.shape, False, dtype=bool)
        else:
            raise ValueError("mask region name {} invalid".format(self.region))
        return mask

    def min_constraint(self):
        try:
            min_val = np.amin(self.constraint)
        except:
            min_val = self.constraint
        return min_val




class Mask1D():

    """
    predefined masks for pixel 1D image data

    Attributes
    ----------
    shape: str
        shape of the mask
    region: str
        together with shape defines where the mask is applied
    constraint: float or np.array<float>(2,)
        one or two constraints for the mask
    """


    def __init__(self, shape, region, constraint, origin=0.):
        """
        Parameters
        ----------
        shape: str
            shape of the mask
        region: str
            together with shape defines where the mask is applied
        constraint: float
            constraint for the mask
        """
        self.shape = shape
        self.region = region
        self.constraint = constraint
        self.current = None
        self.origin = origin

    @classmethod
    def from_mask(Mask1D, other, complement=False):
        if complement:
            region = get_complement(other.region)
        else:
            region = other.region
        new_mask = Mask1D(other.shape, region, np.array(other.constraint))
        return new_mask


    def __repr__(self):
        return "Mask1D({},{},{})".format(self.shape, self.region, self.constraint)

    def get_descriptor(self):
        return np.array([self.shape, self.region, self.constraint])

    def generate_mask(self, x):
        """
        Return a mask array based on shape, region and constraint
        """
        if self.shape == 'window':
            mask = self._get_window_mask(x)
        elif self.shape == None:
            mask = self._get_null_mask(x)
        else:
            raise ValueError("unknown mask type for 1D mask :[{}]".format(self.shape))
        self.current = mask
        return mask

    def _get_null_mask(self, x):
        mask = np.full(x.shape, False, dtype=bool)
        return mask

    def _get_window_mask(self, x):
        x_width = self.constraint
        if self.region == 'center':
            mask = np.abs(x-self.origin) < x_width
        elif self.region =='edge':
            mask = np.abs(x-self.origin) > x_width
        elif self.region =='none':
            mask = np.full(x.shape, False, dtype=bool)
        else:
            raise ValueError("mask region name {} invalid".format(self.region))
        return mask

    def min_constraint(self):
        min_val = self.constraint
        return min_val
