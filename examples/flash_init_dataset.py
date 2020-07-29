from time import time
import xarray as xr
import numpy as np
from glmtools.io.glm import GLMDataset

def get_first_group(group_subset):
    unneeded_dims = ['number_of_flashes', 'number_of_events', 
                     'number_of_field_of_view_bounds', 
                     'number_of_wavelength_bounds', 'number_of_time_bounds']
    
    min_time_idx = group_subset.group_frame_time_offset.argmin().data 
    first_group = group_subset[{'number_of_groups':[min_time_idx]}]
    return first_group.drop_dims(unneeded_dims) 

def flash_init_manual(glm):
    for fl_id, group_subset in glm.parent_groups['group_parent_flash_id']:
        first_group = get_first_group(group_subset)
        # print(first_group)
        print(fl_id, ',', first_group.group_lat.data[0], ',', first_group.group_lon.data[0])
    
def assign_flash_init(ds_one_flash, child_groupby=None):
    assert ds_one_flash.dims['number_of_flashes'] == 1
    
    # get the group dimension indices corresponding to this flash
    groups_this_flash = child_groupby.groups[ds_one_flash.flash_id.data[0]]
    # subset the groups and find the first group
    foo = ds_one_flash[{'number_of_groups':groups_this_flash}]
    first_group = get_first_group(foo)
    ds_one_flash['flash_init_lat'] = xr.DataArray(first_group.group_lat.data,
        dims=['number_of_flashes'])
    ds_one_flash['flash_init_lon'] = xr.DataArray(first_group.group_lon.data,
        dims=['number_of_flashes'])
    return ds_one_flash

def get_flash_init_data(glm):
    init_group_iter = (
        (first_group.group_parent_flash_id.data, first_group.group_lon.data,
                first_group.group_lat.data) for first_group in
        (get_first_group(group_subset) for (fl_id, group_subset) in
        glm.parent_groups['group_parent_flash_id'])
        )
    init_data = np.fromiter(init_group_iter, 
        dtype=[('id', 'i4'), ('lon', 'f4'), ('lat', 'f4')])
    return init_data
    
def add_flash_init_data(glm):
    init_data = get_flash_init_data(glm)
    # use the automatic aligment of coordinate labels to align the data and assign
    # back to the original dataset. The init_id_test variable isn't needed, but 
    # can be used to check the alignment of the data, as shown below.
    glmflidxd=glm.dataset.set_index({'number_of_flashes':'flash_id'})
    init_idx = {'number_of_flashes':init_data['id']}
    init_lon = xr.DataArray(init_data['lon'], dims=['number_of_flashes'],  coords=init_idx)
    init_lat = xr.DataArray(init_data['lat'], dims=['number_of_flashes'],  coords=init_idx)
    init_id_test = xr.DataArray(init_data['id'], dims=['number_of_flashes'],  coords=init_idx)
    glmflidxd['flash_init_lon'] = init_lon
    glmflidxd['flash_init_lat'] = init_lat
    glmflidxd['flash_init_id_test'] = init_id_test
    # make sure the IDs were aligned.
    for v in zip(glmflidxd.number_of_flashes.data,glmflidxd.flash_init_id_test.data):
        assert(int(v[1]-v[0])==0) 
    new_glm = glmflidxd.reset_index('number_of_flashes').rename({'number_of_flashes_':'flash_id'})
    return new_glm
   
def calculate_flash_init(glm):
    new_glm = glm.entity_groups['flash_id'].map(assign_flash_init, 
            child_groupby=glm.parent_groups['group_parent_flash_id'] )
    return(new_glm)


glm=GLMDataset('/Users/ebruning/Downloads/OR_GLM-L2-LCFA_G16_s20200150215200_e20200150215400_c20200150215427.nc')     


# ----- METHOD 0: use the group data to calculate and print the flash IDs and
#                 the lat,lon of their first group.
t0 = time()
print('--- Method 0 ---')
flash_init_manual(glm)
print(time()-t0)


# ----- METHOD 1: as fast as above, but assigns back to the original dataset.
#                 You can use get_flash_init_data if you just want the data
#                 printed above.
t0 = time()
new_glm = add_flash_init_data(glm)
# print(new_glm)
print('--- Method 1 ---')
for flid, ilon, ilat in zip(new_glm.flash_id.data,
        new_glm.flash_init_lat.data, new_glm.flash_init_lon.data):
    print(flid, ',', ilon, ',', ilat)
print(time()-t0)
    
# ----- METHOD 2: 5x slower, but more idiomatically xarray-ish split-apply-merge
print('--- Method 2 ---')
t0 = time()
new_glm = calculate_flash_init(glm)
# print(new_glm)
for flid, ilon, ilat in zip(new_glm.flash_id.data,
        new_glm.flash_init_lat.data, new_glm.flash_init_lon.data):
    print(flid, ',', ilon, ',', ilat)
print(time()-t0)