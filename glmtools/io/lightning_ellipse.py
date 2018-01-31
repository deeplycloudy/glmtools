import numpy as np

def ltg_ellps_lon_lat_to_fixed_grid(lon, lat, sat_lon,
        re_ltg_ellps=6.394140e6, rp_ltg_ellps = 6.362755e6,
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

# Values from the GOES-R L1b PUG Vol 3. These are values *without* the lightnign ellipsoid
# test_lat = 33.846162
# test_lon = -84.690932
# test_alt = 0.0
# test_fixx = -0.024052
# test_fixy = 0.095340
# test_fixz = 0.0
if __name__ == '__main__':    
    goes_lon = -75.0
    test_lon = -110., -110., -57., -57.
    test_lat = 42., -45., 46., -41.
    fix_testx =  np.asarray((-7.072352765696357, -6.693857076039071,  3.593939139528318,  3.945876828443619))*1e4
    fix_testy =  np.asarray((1.106891557528977,  -1.163584191758841,  1.199674272612177, -1.105394721298256))*1e5

    fix_x, fix_y = ltg_ellps_lon_lat_to_fixed_grid(test_lon, test_lat, goes_lon)
    print(fix_x*1e6 - fix_testx)
    print(fix_y*1e6 - fix_testy)