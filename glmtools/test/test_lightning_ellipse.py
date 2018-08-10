from datetime import datetime
import numpy as np

from glmtools.io.lightning_ellipse import (ltg_ellps_lon_lat_to_fixed_grid, 
    ltg_ellps_radii)

# Values from the GOES-R L1b PUG Vol 3. These are values *without* the lightnign ellipsoid
# test_lat = 33.846162
# test_lon = -84.690932
# test_alt = 0.0
# test_fixx = -0.024052
# test_fixy = 0.095340
# test_fixz = 0.0


def test_ellipse_at_launch():
    # This example based on the ellipsoid *at launch*
    goes_lon = -75.0
    test_lon = -110., -110., -57., -57.
    test_lat = 42., -45., 46., -41.
    fix_testx =  np.asarray((-7.072352765696357, -6.693857076039071,  3.593939139528318,  3.945876828443619))*1e4
    fix_testy =  np.asarray((1.106891557528977,  -1.163584191758841,  1.199674272612177, -1.105394721298256))*1e5

    fix_x, fix_y = ltg_ellps_lon_lat_to_fixed_grid(test_lon, test_lat, goes_lon)

    # Differences are < 0.5 microradians
    assert np.allclose(fix_y*1e6, fix_testy)
    assert np.allclose(fix_x*1e6, fix_testx)
    
def test_ellipse_revJ_ellipse():

    date = datetime(2030, 12, 31)
    re, rp = ltg_ellps_radii(date)
    
    # This example based on the ellipsoid *at launch*
    goes_lon = -75.0
    test_lon = -110., -110., -57., -57.
    test_lat = 42., -45., 46., -41.
    fix_testx = np.asarray([-70709.97522351, -66926.99992007, 35933.3006532, 39450.81169228])
    fix_testy = np.asarray([ 110667.78669847, -116338.15403498,  119946.98040115, -110517.06815844])

    fix_x, fix_y = ltg_ellps_lon_lat_to_fixed_grid(test_lon, test_lat, goes_lon,
        re_ltg_ellps=re, rp_ltg_ellps=rp)

    fix_x *= 1e6
    fix_y *= 1e6
    
    # Differences are < 0.5 microradians
    assert np.allclose(fix_y, fix_testy)
    assert np.allclose(fix_x, fix_testx)