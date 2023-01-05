import numpy as np
import scipy
import scipy.spatial

def sum_triangles(xy, z, triangles):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        zsum += area * z[tri].mean(axis=0)
        areasum += area
    return (zsum, areasum)


class Integrator2D():

    def __init__(self, xy, data):
        self.xy = xy
        self.r = np.linalg.norm(xy, axis=1)
        self.data = data
        self.interpolation = None
        self.interp_scheme = 'CloughTocher'

    def integrate_circle(self, radius, strategy='direct'):
        self.integration_radius = radius
        if strategy == 'direct':
            mask = self.r <= radius
            xy = self.xy[mask, :]
            data = self.data[mask]
            return self._integrate_triangulation_linear(xy, data)
        elif strategy == 'interpolation':
            self._interpolate()
            return self._integrate_dblquad()

    def integrate_rectangular(self, constraints):
        self.integration_radius = radius
        if strategy == 'direct':
            mask = self.r <= radius
            xy = self.xy[mask, :]
            data = self.data[mask]
            return self._integrate_triangulation_linear(xy, data)
        elif strategy == 'interpolation':
            self._interpolate()
            return self._integrate_dblquad()



    def _integrate_triangulation_linear(self, xy, data):
        triangulation = scipy.spatial.Delaunay(xy)
        zsum, areasum = sum_triangles(xy, data, triangulation.vertices)
        return zsum/areasum

    def _interpolate(self):
        if self.interp_scheme == 'Linear':
            interp_func = scipy.interpolate.LinearNDInterpolator
        elif self.interp_scheme == 'CloughTocher':
            interp_func = scipy.interpolate.CloughTocher2DInterpolator
        self.interpolation = interp_func(self.xy, self.data)

    def _int_func(self, y, x):
        tmp_array=  np.array([x, y]).reshape(2,-1).T
        tmp = self.interpolation(tmp_array)
        return tmp

    def _upper_circle_limit(self, x):
        r = self.integration_radius
        y = np.sqrt(r**2-x**2)
        return y

    def _lower_circle_limit(self, x):
        r = self.integration_radius
        y = -np.sqrt(r**2-x**2)
        return y


    def _integrate_dblquad(self):
        integration_func = lambda y, x : self._int_func(y, x)
        hfun = lambda x : self._upper_circle_limit(x)
        gfun = lambda x : self._lower_circle_limit(x)
        int_val = scipy.integrate.dblquad(integration_func,
                                          -self.integration_radius,
                                          self.integration_radius,
                                          gfun, hfun)[0]
        return int_val
