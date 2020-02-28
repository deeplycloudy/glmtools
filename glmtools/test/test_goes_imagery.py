from datetime import datetime

import numpy as np

from glmtools.io.imagery import (get_goes_imager_proj, 
    get_goes_imager_fixedgrid_coords, get_glm_global_attrs, 
    get_goes_imager_all_valid_dqf,glm_image_to_var)


def test_get_parts():
    nadir_lon = -75.0
    goes_proj = get_goes_imager_proj(nadir_lon)
    
    # OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231
    start = datetime(2018, 7, 2, 4, 33, 0)
    end = datetime(2018, 7, 2, 4, 34, 0)
    platform = "G16"
    slot = "GOES-East"
    instrument = "GLM-1"
    scene = "CONUS"
    resolution = "2km at nadir"
    timeline = "ABI Mode 3"
    prod_env = "DE"
    prod_src = "Postprocessed"
    prod_site = "TTU"
    
    global_attr = get_glm_global_attrs(start, end, platform, slot, instrument, 
        scene, resolution, timeline, prod_env, prod_src, prod_site, )
    assert global_attr['scene_id'] == scene 
    
    x = np.arange(10, dtype='float')
    y = (np.arange(20, dtype='float')-10)*100
    xm, ym = np.meshgrid(x, y)
    d = xm*ym
    assert d.shape==(20, 10)
    
    dims = ('y', 'x')
    img_var = glm_image_to_var(d, 'Total Energy', 'GLM Total Energy', 'J', dims)
    assert img_var.shape==(20,10)
    
    dqf = get_goes_imager_all_valid_dqf(dims, y.shape+x.shape)
    
    assert dqf.shape[0] == y.shape[0]
    assert dqf.shape[1] == x.shape[0]
    assert (dqf == 0).all()
    assert dqf.encoding['_FillValue']==255
    assert dqf.encoding['_Unsigned']=='true'
    # assert something about unsigned and the fill value being 255
    
    x_var, y_var = get_goes_imager_fixedgrid_coords(x,y)
    assert x_var.units == 'rad'
    assert y_var.units == 'rad'

    assert goes_proj.longitude_of_projection_origin == nadir_lon
    
    