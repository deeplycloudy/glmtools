import os, glob
import numpy as np
from glmtools.io.glm import GLMDataset
import xarray

def get_four_level_data():
    storm_id = [0,1,2] 
    flash_parent_storm_id = [0,0,0,2,2,2,2,2]
    flash_id =              [1,2,3,4,5,6,7,8]
    stroke_parent_flash_id = [1,1,2,3, 4, 4, 4, 6, 8, 8, 8]
    stroke_id =              [4,6,7,9,13,14,15,19,20,23,46]
    trig_parent_stroke_id = [4,4,4,4,6, 7, 7, 9,13,13,14,14,15,19,20,20,23,46]
    trig_id =               [1,3,5,8,9,10,12,16,18,19,20,22,23,25,26,30,31,32]
    trig_parent_storm_id =  [0,0,0,0,0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    trig_parent_flash_id =  [1,1,1,1,1, 2, 2, 3, 4, 4, 4, 4, 4, 6, 8, 8, 8, 8]

    storm_child_flash_count = [3,0,5]
    flash_child_stroke_count = [2,1,1,3,0,1,0,3]
    stroke_child_trig_count = [4,1,2,1,2,2,1,1,2,1,1]
    
    storm_child_stroke_count = [4,0,7]
    storm_child_trig_count = [8,0,10]
    
    storm_dim = 'nstorms'
    flash_dim = 'nflashes'
    stroke_dim = 'nstrokes'
    trig_dim = 'ntrigs'
    
    d = xarray.Dataset({
        'storm_id': (storm_dim, storm_id),
        'flash_id': (flash_dim, flash_id),
        'flash_parent_storm_id': (flash_dim, flash_parent_storm_id),
        'stroke_id': (stroke_dim, stroke_id),
        'stroke_parent_flash_id': (stroke_dim, stroke_parent_flash_id),
        'trig_id': (trig_dim, trig_id),
        'trig_parent_stroke_id': (trig_dim, trig_parent_stroke_id),
        'trig_parent_flash_id': (trig_dim, trig_parent_flash_id),
        'trig_parent_storm_id': (trig_dim, trig_parent_storm_id),
        'storm_child_flash_count': (storm_dim, storm_child_flash_count),
        'storm_child_stroke_count': (storm_dim, storm_child_stroke_count),
        'storm_child_trig_count': (storm_dim, storm_child_trig_count),
        'flash_child_stroke_count': (flash_dim, flash_child_stroke_count),
        'stroke_child_trig_count': (stroke_dim, stroke_child_trig_count),
        })
    d = d.set_coords(['stroke_id', 'flash_id', 'storm_id', 'trig_id', 
                      'stroke_parent_flash_id', 'trig_parent_stroke_id', 
                      'flash_parent_storm_id'])
    assert len(flash_id) == len(flash_parent_storm_id)
    assert len(stroke_id) == len(stroke_parent_flash_id)
    assert len(trig_id) == len(trig_parent_stroke_id)
    assert sum(storm_child_flash_count) == len(flash_id)
    assert sum(storm_child_stroke_count) == len(stroke_id)
    assert sum(storm_child_trig_count) == len(trig_id)
    assert sum(flash_child_stroke_count) == len(stroke_id)
    assert sum(stroke_child_trig_count) == len(trig_id)
    return d

def get_sample_data_path():
    import glmtools.test
    data_path = os.path.abspath(glmtools.test.__path__[0])
    data_folder = 'data' #inside data_path
    path = os.path.join(data_path, data_folder)
    return path

def get_sample_data_list():
    path = get_sample_data_path()
    filenames  = glob.glob(os.path.join(path, '*.nc'))
    return filenames
    
def get_test_dataset():
    # filename ='/data/LCFA-production/OR_GLM-L2-LCFA_G16_s20171161230400_e20171161231000_c20171161231027.nc'
    # flash_ids=np.array([6359, 6472, 6666])
    path = get_sample_data_path()
    filename = os.path.join(path, 'FGE_split_merge_GLM.nc')
    flash_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    flash_ids.sort()
    glm = GLMDataset(filename)
    return glm, flash_ids
