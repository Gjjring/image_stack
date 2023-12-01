import numpy as np
from typing import Union
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

def mask_regions_for_shape(shape):
    mask_regions = {}
    mask_regions['window'] = ['all', 'center', 'edge', 'left', 'right']
    mask_regions['circular'] = ['all', 'center', 'edge']
    mask_regions['rectangular'] = ['all', 'center', 'edge',
                                    'horizontal', 'vertical',
                                    'north', 'south', 'east', 'west']
    mask_regions[None] = [None]
    if shape == 'None':
        shape = None
    if shape not in mask_regions:
        raise KeyError("Unknown mask shape: [{}]".format(shape))
    return mask_regions[shape]

def reduce_region_dimension(region):
    reduced_regions = {}
    reduced_regions[None] = [None]
    reduced_regions['all'] = ['all']
    reduced_regions['center'] = ['center']
    reduced_regions['edge'] = ['horizontal', 'vertical']
    reduced_regions['left'] = ['south', 'west']
    reduced_regions['right'] = ['north', 'east']

    for key, value in reduced_regions.items():
        for sub_value in value:
            if sub_value == region:
                return key
    raise ValueError("could not reduce region: {}".format(region))


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


    def __init__(self, shape: str, constraint: Union[float, np.ndarray],
                 region: str='all',
                 origin: np.ndarray=np.array([0., 0.]),
                 complement: bool=False,
                 region_constraint: float=0.,
                 tolerance: float=1e-6) -> 'Mask2D':
        """
        Parameters
        ----------
        shape: str
            shape of the mask
        constraint: float
            constraint for the mask
        region: str
            sub category of shape
        origin: float
            mask origin
        complement: bool
            mask values outside of constraint
        region_constraint: float
            width of mask for certain regions
        """
        if shape not in ['circular', 'rectangular', 'None', None]:
            raise ValueError("unknown shape for 2D mask: [{}]".format(shape))
        if shape == 'None':
            shape = None
        self.shape = shape
        regions = mask_regions_for_shape(shape)
        if region == 'None':
            region = None
        if not region in regions:
            raise ValueError("unknown region for {} mask: [{}]".format(shape, region))
        self.region = region
        self.constraint = constraint
        self.origin = origin
        self.region_constraint = region_constraint
        self.complement = complement
        self.tolerance = tolerance
        self.current = None
        self.polar = False


    @classmethod
    def from_mask(Mask2D, other: 'Mask2D') -> 'Mask2D':
        new_mask = Mask2D(other.shape, np.array(other.constraint),
                          region=other.region,
                          origin=np.array(other.origin),
                          complement=other.complement,
                          region_constraint=other.region_constraint,
                          tolerance=other.tolerance)
        new_mask.polar = other.polar
        return new_mask

    def __repr__(self):
        return "Mask2D({},{},{},{},{})".format(self.shape, self.region, self.constraint, self.origin, self.region_constraint)

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
        elif self.shape is None or self.shape == 'None':
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
        if self.region == 'all':
            mask = R > self.constraint
        elif self.region == 'center':
            mask = R > self.region_constraint
        elif self.region == 'edge':
            mask_outer_edge = R > self.constraint
            mask_inner_edge = R < self.constraint-self.region_constraint
            mask = np.logical_or(mask_outer_edge, mask_inner_edge)
        if self.complement:
            mask = np.logical_not(mask)
        return mask

    def _get_rectangular_mask(self, X, Y):
        xy_width = self.constraint
        mask_x = np.abs(X-self.origin[0]) > xy_width[0]*(1+self.tolerance)
        mask_y = np.abs(Y-self.origin[1]) > xy_width[1]*(1+self.tolerance)
        all_mask = np.logical_or(mask_x, mask_y)
        if self.region == 'all':
            mask = all_mask
        elif self.region == 'center':
            mask_x = np.abs(X-self.origin[0]) > self.region_constraint*(1+self.tolerance)
            mask_y = np.abs(Y-self.origin[1]) > self.region_constraint*(1+self.tolerance)
            mask = np.logical_or(mask_x, mask_y)
        elif self.region == 'edge':
            mask2_x = np.abs(X-self.origin[0]) > xy_width[0]-self.region_constraint*(1+self.tolerance)
            mask2_y = np.abs(Y-self.origin[1]) > xy_width[1]-self.region_constraint*(1+self.tolerance)
            mask2 = np.logical_not(np.logical_or(mask2_x, mask2_y))
            mask = np.logical_or(all_mask, mask2)
        elif self.region == 'horizontal':
            mask2_x = np.abs(X-self.origin[0]) > xy_width[0]-self.region_constraint*(1+self.tolerance)
            mask2_x = np.logical_not(mask2_x)
            mask = np.logical_or(all_mask, mask2_x)
        elif self.region == 'vertical':
            mask2_y = np.abs(Y-self.origin[1]) > xy_width[1]-self.region_constraint*(1+self.tolerance)
            mask2_y = np.logical_not(mask2_y)
            mask = np.logical_or(all_mask, mask2_y)
        elif self.region == 'north':
            mask2_y = Y-self.origin[1] > xy_width[1]-self.region_constraint*(1+self.tolerance)
            mask2_y = np.logical_not(mask2_y)
            mask = np.logical_or(all_mask, mask2_y)
        elif self.region == 'south':
            mask2_y = -(Y-self.origin[1]) > xy_width[1]-self.region_constraint*(1+self.tolerance)
            mask2_y = np.logical_not(mask2_y)
            mask = np.logical_or(all_mask, mask2_y)
        elif self.region == 'east':
            mask2_x = -(X-self.origin[0]) > xy_width[0]-self.region_constraint*(1+self.tolerance)
            mask2_x = np.logical_not(mask2_x)
            mask = np.logical_or(all_mask, mask2_x)
        elif self.region == 'west':
            mask2_x = X-self.origin[0] > xy_width[0]-self.region_constraint*(1+self.tolerance)
            mask2_x = np.logical_not(mask2_x)
            mask = np.logical_or(all_mask, mask2_x)
        else:
            raise ValueError("unknown region: {}".format(self.region))
        return mask

    def min_constraint(self):
        try:
            min_val = np.amin(self.constraint)
        except:
            min_val = self.constraint
        return min_val

    def convert_to_1D(self, dimension: str) -> 'Mask1D':
        if self.shape is None:
            return Mask1D(None, 0.)

        if dimension == 'x':
            origin = self.origin[1]
        elif dimension =='y':
            origin = self.origin[0]
        else:
            raise ValueError("dimension must be x or y not {}".format(dimension))
        reduced_region = reduce_region_dimension(self.region)
        if self.shape == 'rectangular':
            if dimension == 'x':
                constraint = self.constraint[1]
            elif dimension == 'y':
                constraint = self.constraint[0]
            return Mask1D('window', constraint, region=reduced_region,
                          origin=origin, complement=self.complement,
                          region_constraint=self.region_constraint,
                          tolerance=self.tolerance)
        elif self.shape == 'circular':
            constraint = self.constraint
            return Mask1D('window', constraint, region=reduced_region,
                          origin=origin, complement=self.complement,
                          region_constraint=self.region_constraint,
                          tolerance=self.tolerance)





class Mask1D():

    """
    predefined masks for pixel 1D image data

    Attributes
    ----------
    shape: str
        shape of the mask
    constraint: float
        mask width
    region: str
        sub category of shape
    origin: float
        mask origin
    complement: bool
        mask values outside of constraint
    """


    def __init__(self, shape: str, constraint: float, region: str='all',
                 origin: float=0., complement: bool=False,
                 region_constraint: float=0.,
                 tolerance: float=1e-6) -> 'Mask1D':
        """
        Parameters
        ----------
        shape: str
            shape of the mask
        constraint: float
            constraint for the mask
        region: str
            sub category of shape
        origin: float
            mask origin
        complement: bool
            mask values outside of constraint
        region_constraint: float
            width of mask for certain regions
        """
        if shape not in ['window', 'None', None]:
            raise ValueError("unknown shape for 1D mask: [{}]".format(shape))
        if shape == 'None':
            shape = None
        self.shape = shape
        regions = mask_regions_for_shape(shape)
        if region == 'None':
            region = None
        if region not in regions:
            raise ValueError("unknown region for {} mask: [{}]".format(shape, region))
        self.region = region
        self.constraint = constraint
        self.origin = origin
        self.region_constraint = region_constraint
        self.complement = complement
        self.tolerance = tolerance
        self.current = None

    @classmethod
    def from_mask(Mask1D, other: 'Mask1D') -> 'Mask1D':
        new_mask = Mask1D(other.shape, np.array(other.constraint),
                          region=other.region,
                          origin=np.array(other.origin),
                          complement=other.complement,
                          region_constraint=other.region_constraint,
                          tolerance=other.tolerance)
        return new_mask


    def __repr__(self) -> str:
        return "Mask1D({},{},{},{},{})".format(self.shape, self.region, self.constraint, self.origin, self.region_constraint)

    def get_descriptor(self) -> np.ndarray:
        return np.array([self.shape, self.region, self.constraint, self.origin])

    def generate_mask(self, x:np.ndarray) -> np.ndarray:
        """
        Return a mask array based on shape, region and constraint
        """
        if self.shape == 'window':
            mask = self._get_window_mask(x)
        elif self.shape is None or self.shape == 'None':
            mask = self._get_null_mask(x)
        else:
            raise ValueError("unknown mask type for 1D mask :[{}]".format(self.shape))
        self.current = mask
        return mask

    def _get_null_mask(self, x:np.ndarray) -> np.ndarray:
        mask = np.full(x.shape, False, dtype=bool)
        return mask

    def _get_window_mask(self, x:np.ndarray) -> np.ndarray:
        if self.region == 'all':
            mask = np.abs(x-self.origin) > self.constraint*(1+self.tolerance)
        elif self.region == 'center':
            mask = np.abs(x-self.origin) > self.region_constraint*(1+self.tolerance)
        elif self.region == 'left':
            mask = np.abs(x-(self.origin-self.constraint+0.5*self.region_constraint)) > 0.5*self.region_constraint*(1+self.tolerance)
        elif self.region == 'right':
            mask = np.abs(x-(self.origin+self.constraint-0.5*self.region_constraint)) > 0.5*self.region_constraint*(1+self.tolerance)
        elif self.region == 'edge':
            mask_left = np.abs(x-(self.origin+self.constraint-0.5*self.region_constraint)) > 0.5*self.region_constraint*(1+self.tolerance)
            mask_right = np.abs(x-(self.origin-self.constraint+0.5*self.region_constraint)) > 0.5*self.region_constraint*(1+self.tolerance)
            mask = np.logical_and(mask_left, mask_right)
        else:
            raise ValueError("cannot get window mask for region: {}".format(self.region))
        if self.complement:
            mask = np.logical_not(mask)
        return mask

    def min_constraint(self) -> float:
        min_val = self.constraint
        return min_val
