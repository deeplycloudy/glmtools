"""

This module contains utilities for converting 2D arrays of longitude and
latitude of mean GLM CCD pixel positions into a lookup table which, given
arbitrary event latitude and longitude, may be used to quickly find the pixel
ID of that event. 

This supports two crucial needs:
- Determining the pixel adjacency information for any given flash so that
  the flash or group footprint can be plotted as contiguous pixels instead
  of points, on a precise per-flash basis.
    - Requires an algorithm to create edges of pixel grids using navigated
      L2 event locations. The lookup table is used to infer the pixel geometry/
      adjacency, with the pixel locations inferred from the L2 locations.
- Creating a fixed grid with spacing that matches the varying pixel size
  across relatively large domains, suitable for imagery-like loops of 
  event density, flash extent density, etc.
    - Simply requires a subset of the 2D mean lat/lon grid for the domain
      of interest.

This approach will give the exact CCD pixel location if the navigated pixel
jitter is less than half the width of that pixel.

In the real case, there may be navigation corrections greater than one pixel,
resulting in a possible multiple pixel offset from the true CCD (x,y) position. 
However, such concerns are moot for the purposes of a GLM fixed grid as needed 
for certain research and most operational uses of the GLM.

The location of events on the fixed grid will never be misplaced by more than 
1/2 pixel. This is because because the fixed grid is defined in a way that 
matches the GLM pixel spacing, the navigated events used to populate the fixed 
grid retain this spacing, and the pixel spacing does not vary greatly on the 
scale of a few pixels. Therefore, some subpixel jitter may occur, but over many 
flashes the locations of storm cores would still be accurate.

It would be possible to reintroduce multi-pixel jitter if this approach were 
used to find the the pixel ID, and that pixel ID were re-navigated using the
satellite orientation parameters as though the CCD id were exact. However,
there is no need to perform this step; one would presumably start with L0 data
if the goal were to re-navigate the data from the satellite pointing data.

"""


import pickle
import tables
import numpy as np
from sklearn.neighbors import KDTree

def read_pixel_location_mat_file(filename, lat_var='mean_lat', lon_var='mean_lon'):
    """ 
    
    Read a .mat file containing 2D arrays of pixel center latitude and
    longitude values. The optional keyword arguments may be used to change the
    variable names to read.
    
    The .mat file is assumed to be a modern, HDF5-based .mat file.
    
    Returns:
    lons, lats: 2D masked arrays of longitude and latitude.
    """
    nav = tables.open_file('GLM_base.mat')
    lats = getattr(nav.root, lat_var)[:]
    lons = getattr(nav.root, lon_var)[:]
    lats = np.ma.masked_array(lats, np.isnan(lats))
    lons = np.ma.masked_array(lons, np.isnan(lons))-360
    return lons, lats

def create_pixel_lookup(lons, lats, leaf_size=40):
    """
    lons: 2D array of pixel center longitudes. Shape is (1300, 1372) for GLM.
    lats: 2D array of pixel center latitudes. Shape (1300, 1372) for GLM.
    
    Pixel ids are defined such that:
    x increases along the first dimension (1300) from 0 ... Nx-1
    y increases along the second dimension (1372) from 0 ... Ny-1
    Therefore, the pixel ids may be used to index lons, lats.
    
    If there are bad or missing values, lons and lats are expected to be numpy
    masked arrays.
    
    Returns: lookup, X, Y
    lookup is an instance of sklearn.neighbors.KDTree 
    
    
    The pixel ids for arbitrary locations may be found as follows. This example
    also shows how to find the lons and lats of the pixel centers:
    
    >>> test_events = np.vstack([(33.5, -101.5), (32.5, -102.8), (32.5, -102.81)])
    >>> distances, idx = lookup.query(test_events)
    >>> # Use the pixel IDs to query the lon,
    >>> loni, lati = lons[X[idx], Y[idx]], lats[X[idx], Y[idx]]
        
    """
    
    x = np.arange(lons.shape[0])
    y = np.arange(lons.shape[1])
    
    if ((type(lats) == np.ma.core.MaskedArray) |  
        (type(lons) == np.ma.core.MaskedArray)):
        good = ~(lats.mask | lons.mask)
    else:
        # all are assumed good
        good = np.ones_like(lats, dtype=bool)
    X, Y = np.meshgrid(x, y)
    Xgood = X.T[good].flatten()
    Ygood = Y.T[good].flatten()

    flatlat = lats[good].flatten()
    flatlon = lons[good].flatten()
    flat_geo = np.vstack((flatlon, flatlat)).T
    # Benchmark using approach in https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/
    lookup = KDTree(flat_geo, leaf_size=leaf_size)
    return lookup, Xgood, Ygood

def save_pixel_lookup(lookup, X, Y, lons, lats, filename=None):
    """ 
    Save the pixel lookup information and the corresponding pixel IDs
    (as produced by create_pixel_lookup, plus the input lons and lats to that
    function) to filename, or if filename is not provided return a byte string 
    representing the pickle.
    """
    obj = (lookup, X, Y, lons, lats)
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    else:
        return pickle.dumps(obj)

def load_pixel_lookup(filename):
    """
    Returns (lookup, X, Y), the pickeled objects created by save_pixel_lookup.
    Equivalent to the arguments returned by create_pixel_lookup.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


