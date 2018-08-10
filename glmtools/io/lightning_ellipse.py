from datetime import datetime
import numpy as np

def ltg_ellps_radii(date):
    """
    Given a date, return the equatorial and polar radii of the lightning
    ellipsoid. The ellipsoid was tuned after launch based on comparison of the
    GLM flash centroids to ground strike locations.
    
    date: datetime object for the date and time of observation
    
    Returns: re_ltg_ellps, rp_ltg_ellps (meters): equatorial and polar radii
        of the lightning ellipsoid, respectively.
    """
    if date < datetime(2018,10,9):
        re_ltg_ellps, rp_ltg_ellps = 6.394140e6, 6.362755e6
    else:
        # The GRS80 altitude + 6 km differs by about 3 m from the value above
        # which is the exact that was provided at the time of launch. Use the
        # original value instead of doing the math.
        # 6.35675231414e6+6.0e3
        re_ltg_ellps, rp_ltg_ellps = 6.378137e6 + 14.0e3, 6.362755e6
    return re_ltg_ellps, rp_ltg_ellps

def ltg_ellps_lon_lat_to_fixed_grid(lon, lat, sat_lon, date,
        re_grs80 = 6.378137e6, rp_grs80 = 6.35675231414e6,
        sat_grs80_height=35.786023e6 ):
    """ 
    lon, lat (degrees): from GLM L2 file, fixed grid coords x, y as 
    defined in the L1b PUG. x,y corresponds to beta, alpha.
        
    sat_lon (degrees): nominal nadir longitude of the satellite

    re_ltg_ellps, rp_ltg_ellps (meters): equatorial and polar radii 
        for the lightning ellipse. Defaults to values set at launch
        of GOES-16 GLM (good at least through early Jan 2018).

    sat_grs80_height (meters): height of the satellite above the GRS80 
        ellipsoid. This is 'perspective_point_height' in the GOES-R L1b PUG, 
        and the attribute of the same name in the goes_imager_projection 
        variable.

    re_grs80, rp_grs80 (meters): equatorial and polar radii 
        for GRS80 lightning ellipse used by GOES-R, as defined in the
        GOES-R L1b PUG, and the semi_major_axis and semi_minor_axis
        attributes of the goes_imager_projection variable.

    This function undoes the lightning ellipsoid height assumption,
    such that the final fixed grid position matches the ABI fixed
    grid definition and therefore the ABI L1b products.

    Reference:
    Bezooijen, R. W. H., H. Demroff, G. Burton, D. Chu, and S. Yang, 2016: 
        Image navigation and registration for the geostationary lightning 
        mapper (GLM). Proc. SPIE 10004, 100041N, doi: 10.1117/12.2242141.
    """
    re_ltg_ellps, rp_ltg_ellps = ltg_ellps_radii(date)

    ff_ltg_ellps = (re_ltg_ellps - rp_ltg_ellps)/re_ltg_ellps
    ff_grs80 = (re_grs80 - rp_grs80)/re_grs80 # 0.003352810704800 
    sat_H = sat_grs80_height + re_grs80 # 42.164e6 
    
    # center longitudes on satellite, and ensure between +/- 180
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    dlon = lon-sat_lon
    dlon[dlon < -180] += 360
    dlon[dlon > 180] -= 360
    lon_rad = np.radians(dlon)
    lat_rad = np.radians(lat)

    lat_geocent = np.arctan( (1.0 - ff_grs80)**2.0 * np.tan(lat_rad))
    
    # We assume geocentric latitude
    cos_factor = np.cos(lat_geocent)
    sin_factor = np.sin(lat_geocent)

    R = re_ltg_ellps*(1-ff_ltg_ellps) / np.sqrt(1.0-ff_ltg_ellps*(2.0-ff_ltg_ellps)*cos_factor*cos_factor)
    vx = R * cos_factor * np.cos(lon_rad) - sat_H
    vy = R * cos_factor * np.sin(lon_rad)
    vz = R * sin_factor
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)
    vx /= -vmag # minus signs flip so x points to earth, z up, y left
    vy /= -vmag
    vz /= vmag
    
    # Microradians
    alpha = np.arctan(vz/vx) #* 1e6
    beta = -np.arcsin(vy) #* 1e6
    return beta, alpha
