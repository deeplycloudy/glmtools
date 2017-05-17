import itertools

import numpy as np
import xarray as xr

def fix_unsigned(data, is_xarray=True):
    """
    The function is used to fix data written as signed integer bytes
    with an "_Unsigned" attribute, but which have been automatically
    converted to floating point by xarray. 
    
    This function removes the scale_factor and add_offset, and then casts
    to an unsigned integer type, before scaling the data once again.
    
    Returns a 64-bit numpy floating point array. 
    
    Could be used to rewrite fix_event_locations, but should be equivalent.
    """
    dtype = data.encoding['dtype'].str.replace('i', 'u')
    scale_factor = data.encoding['scale_factor']
    add_offset = data.encoding['add_offset']
    unscale = ((data - add_offset)/scale_factor).data.astype(dtype).astype('float64')
    fixed = unscale * scale_factor + add_offset
    return fixed

def fix_event_locations(event_lats, event_lons, is_xarray=False):
    """ event_lats and event_lons are netCDF4 Variables, or
        if is_xarray=True, xarray Variables.
    
        returns fixed (event_lats, event_lons)
    
        This function is used to correct for the NetCDF-Java convention of writing 
        signed int16 and tagging it with an _Unsigned attribute. Per 
        http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/Netcdf4.html
        NetCDF Java cannot write a proper unsigned integer.
    
        If the version of netcdf4-python used to read the data is >=1.2.8, then 
        this function is not needed. The issue corrected by this function was 
        built into netcdf4-python in PR #658 developed in response to issue #656. 
    
        xarray turns off auto-scaling, which also turns off the unsigned int
        correction in netCDF4-python. xarray then applies the scale/offset itself.
        Therefore, this function is still needed to undo xarray's scale/offset, 
        followed by unsigned int conversion and reapplication of the scale/offset.
        
        and so the scaling must be worked around 

    """
    
    # From PUG spec, and matches values in file.
    lon_fov = (-156.06, -22.94)
    dlon_fov = lon_fov[1]-lon_fov[0]
    lat_fov = (-66.56, 66.56)
    scale_factor = 0.00203128

    
    if is_xarray==True:
        # unscale the data 
        unscale_lat = ((event_lats - lat_fov[0])/scale_factor).data.astype('int32')
        unscale_lon = ((event_lons - lon_fov[0])/scale_factor).data.astype('int32')
        event_lats = unscale_lat
        event_lons = unscale_lon
    else: # is NetCDF Variable
        event_lons.set_auto_scale(False)
        event_lats.set_auto_scale(False)
        event_lons = event_lons[:].astype('int32')
        event_lats = event_lats[:].astype('int32')

    unsigned = 2**16
    event_lons[event_lons < 0] += unsigned
    event_lats[event_lats < 0] += unsigned
    
    event_lons_fixed = (event_lons)*scale_factor+lon_fov[0]
    event_lats_fixed = (event_lats)*scale_factor+lat_fov[0]
    
    return event_lats_fixed, event_lons_fixed

# These variables are tagged with an "_Unsigned" attribute
glm_unsigned_float_vars = (
'event_lat',
'event_lon',
'event_energy',
'group_area',
'group_energy',
'group_quality_flag',
'flash_area',
'flash_energy',
'flash_quality_flag',
'yaw_flip_flag',
)
glm_unsigned_vars = glm_unsigned_vars + (
'event_id',
'group_id',
'flash_id',
'event_parent_group_id',
'group_parent_flash_id',
)

class GLMDataset(object):
    def __init__(self, filename):
        """ filename is any data source which works with xarray.open_dataset """
        self.dataset = xr.open_dataset(filename)

        self.fov_dim = 'number_of_field_of_view_bounds'
        self.wave_dim = 'number_of_wavelength_bounds'
        self.time_dim = 'number_of_time_bounds'
        self.gr_dim = 'number_of_groups'
        self.ev_dim = 'number_of_events'
        self.fl_dim = 'number_of_flashes'

        self.split_flashes = self.dataset.groupby('flash_id')
        self.split_groups = self.dataset.groupby('group_id')
        self.split_events = self.dataset.groupby('event_id')
        self.split_groups_parent = self.dataset.groupby('group_parent_flash_id')
        self.split_events_parent = self.dataset.groupby('event_parent_group_id')
        
        if len(getattr(self.dataset, self.fl_dim)) > 0:
            self.energy_min, self.energy_max = 0, self.dataset.flash_energy.max()
        else:
            # there are no flashes
            pass

        # for k, v in self.split_groups_parent.groups.items():
        #     print k,v

    @property
    def fov_bounds(self):
#         lat_bnd = self.dataset.lat_field_of_view_bounds.data
#         lon_bnd = self.dataset.lon_field_of_view_bounds.data
        lat_bnd = self.dataset.event_lat.min().data, self.dataset.event_lat.max().data
        lon_bnd = self.dataset.event_lon.min().data, self.dataset.event_lon.max().data
        return lon_bnd,lat_bnd

        
    def lonlat_subset(self, lon_range=None, lat_range=None):
        """ Subset the dataset based on longitude, latitude bounding box. 
            
            Applies subsetting only the the flashes, and then retrieves all events and
            groups that go with those flash ids. 
            
            If a flash's centroid is within the bounding box by has events that straddle
            the bounding box edge, all events will still be returned. Same goes for
            groups. Therefore, the group and event locations are not strictly within 
            the bounding box.
        """
        if lon_range is None:
            lon_range = (-180., 180.)
        if lat_range is None:
            lat_range = (-90., 90.)
            
        good = ((self.dataset.flash_lat < lat_range[1]) & 
                (self.dataset.flash_lat > lat_range[0]) &
                (self.dataset.flash_lon < lon_range[1]) & 
                (self.dataset.flash_lat > lon_range[0])
                ).data
        flash_ids = self.dataset.flash_id[good].data
        return self.get_flashes(flash_ids)
    
    def flash_id_for_events(self, flash_data):
        """ Retrieve an array of flash_ids for each event in flash_data 
            by walking up the group to flash tree. The number of flash ids is
            a equal to the total number of events in flash_data.
        
            flash_data is a (possible) subset of the whole glm.dataset which 
            contains event, group, and flash data.
        
            returns event_parent_flash_id.
        
            to add event_parent_flash_id to the original flash_data:
            flash_data['event_parent_flash_id']=xarray.DataArray(flash_ids_per_event, dims=[self.ev_dim])
        """
        gr_idx = [self.split_groups.groups[gid] for gid in flash_data.event_parent_group_id.data]
        
        # indexing directly with gr_idx doesn't work ... gives an error on a conflicting number of dimensions
        # This might/will likely return nonsense if the group index is not identical 
        # to 0 ... N-1 for the number of elements in group_parent_flash_id, i.e., the index
        # is automatically generated as an arange.    
        flash_ids_per_event = self.dataset.group_parent_flash_id[:].data[gr_idx].squeeze()
        
        # Should be equiavlent to this oneliner, i.e., 
        # (flash_ids_per_event == flash_ids_per_event_careful).all() is true
        # flash_ids_per_event_careful= np.asarray([glm.dataset.group_parent_flash_id[glm.split_groups.groups[gid]]['group_parent_flash_id'].data[0] for gid in  glm.dataset.event_parent_group_id.data])
        # Which extracts data from this version which returns a series of one-group datasets
        # [glm.dataset.group_parent_flash_id[glm.split_groups.groups[gid]] for gid in  glm.dataset.event_parent_group_id.data]

        return flash_ids_per_event

    
    def get_flashes(self, flash_ids):
        """ Subset the dataset to a some flashes with ids given by a list of
            flash_ids. Can be used to retrieve a single flash by passing a
            single element list.
        """
        
        # The list of indices returned by the group dictionary correspond to the 
        # indices automatically generated by xarray for each dimension, and
        # don't correspond to the flash_id and group_id columns
        fl_iter = (self.split_flashes.groups[fid] for fid in flash_ids)
        fl_idx = list(itertools.chain.from_iterable(fl_iter))
        gr_iter = (self.split_groups_parent.groups[fid] for fid in flash_ids)
        gr_idx = list(itertools.chain.from_iterable(gr_iter))
        
        #get just these flashes and their groups
        grp_sub = self.dataset[{self.fl_dim:fl_idx,
                                self.gr_dim:gr_idx
                               }]

        # get event ids for each group and chain them together into one list
        ev_iter = (self.split_events_parent.groups[gid] for gid in grp_sub.group_id.data)
        ev_idx = list(itertools.chain.from_iterable(ev_iter))

        # get just the events that correspond to this flash
        these_flashes = grp_sub[{self.ev_dim:ev_idx}]
        return these_flashes
        

