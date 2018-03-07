import itertools
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from glmtools.io.traversal import OneToManyTraversal
from glmtools.io.lightning_ellipse import ltg_ellps_lon_lat_to_fixed_grid

def parse_glm_filename_time(time_str):
        """ Given an input time string like
            s20171880000200
            e20171880000400
            c20171880000426
            parse the time, including the tenths of a second at the end.
        """
        # The filename convention for julian day is day of year, 001-366
        # This matches the strptime convention. See Appendix A to GOES-R PUG
        # Volume 5.
        start = datetime.strptime(time_str[1:-1], '%Y%j%H%M%S')
        start_tenths = int(time_str[-1])
        start += timedelta(0, start_tenths/10.0)
        return start

def parse_glm_filename(filename):
    """ Parse the GLM filename, reutrning (`ops_environment`, `algorithm`, 
            `platform`, `start`, `end`, `created`)
        The last three values returned are datetime objects.
        
        See Appendix A to the GOES-R PUG Volume 5 for the filename spec.
        OR_GLM-L2-LCFA_G16_s20171880000200_e20171880000400_c20171880000426.nc
    """
    parts = filename.replace('.nc', '').split('_')
    ops_environment = parts[0]
    algorithm = parts[1]
    platform = parts[2]        
    start = parse_glm_filename_time(parts[3])
    end = parse_glm_filename_time(parts[4])
    created = parse_glm_filename_time(parts[5])
    return ops_environment, algorithm, platform, start, end, created
        
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
glm_unsigned_vars = glm_unsigned_float_vars + (
'event_id',
'group_id',
'flash_id',
'event_parent_group_id',
'group_parent_flash_id',
)

def event_areas(flash_data):
    """ Given `flash_data`, which may be a subset of a `GLMDataset.dataset`,
        calculate the area for each event in `flash_data`. The calculation is:
    
        `event_area` = `group_area` / `n_events`    
    """
    event_count = flash_data.group_child_event_count
    group_area = flash_data.group_area
    event_area = event_count / group_area
    # need logic here to replicate the event_area to all events.

class GLMDataset(OneToManyTraversal):
    def __init__(self, filename, calculate_parent_child=True):
        """ filename is any data source which works with xarray.open_dataset
            
            By default, helpful additional parent-child data are calculated,
                'event_parent_flash_id'
                'flash_child_group_count'
                'flash_child_event_count'
                'group_child_event_count'
            and a MultiIndex is set on the dataset.
            Setting `calculate_parent_child=False` avoids the traversal of the
            dataset needed to computate these values. This can be useful when
            one only needs to grab an attribute or two (such as the 
            `product_time`) from the original files. In this state, the only 
            safe route is to access `self.dataset` directly. Other methods of 
            this class are not guaranteed to work.
        """
        dataset = xr.open_dataset(filename)
        self._filename = filename

        self.fov_dim = 'number_of_field_of_view_bounds'
        self.wave_dim = 'number_of_wavelength_bounds'
        self.time_dim = 'number_of_time_bounds'
        self.gr_dim = 'number_of_groups'
        self.ev_dim = 'number_of_events'
        self.fl_dim = 'number_of_flashes'

        idx = {self.gr_dim: ['group_parent_flash_id', 'group_id',
                             'group_time_offset',
                             'group_lat', 'group_lon'],
               self.ev_dim: ['event_parent_group_id', 'event_id',
                             'event_time_offset',
                             'event_lat', 'event_lon'],
               self.fl_dim: ['flash_id',
                             'flash_time_offset_of_first_event',
                             'flash_time_offset_of_last_event',
                             'flash_lat', 'flash_lon']}
                             
        self.entity_ids = ['flash_id', 'group_id', 'event_id']
        self.parent_ids = ['group_parent_flash_id', 'event_parent_group_id']

        if calculate_parent_child:
            # sets self.dataset
            super().__init__(dataset.set_index(**idx), 
                             self.entity_ids, self.parent_ids)
            self.__init_parent_child_data()
        else:
            self.dataset = dataset
    
    def __init_parent_child_data(self):
        """ Calculate implied parameters that are useful for analyses
            of GLM data.
        """
        flash_ids = self.replicate_parent_ids('flash_id', 
                                              'event_parent_group_id'
                                              )
        event_parent_flash_id = xr.DataArray(flash_ids, dims=[self.ev_dim,])
        self.dataset['event_parent_flash_id'] = event_parent_flash_id


        all_counts = self.count_children('flash_id', 'event_id')
        flash_child_count = all_counts[0]
        flash_child_group_count = xr.DataArray(flash_child_count, 
                                               dims=[self.fl_dim,])
        self.dataset['flash_child_group_count'] = flash_child_group_count

        group_child_count = all_counts[1]
        group_child_event_count = xr.DataArray(group_child_count, 
                                               dims=[self.gr_dim,])
        self.dataset['group_child_event_count'] = group_child_event_count
        
        # we can use event_parent_flash_id to get the flash_child_event_count
        # need a new groupby on event_parent_flash_id
        # then count number of flash_ids that match in the groupby
        # probably would be a good idea to add this to the traversal class
        grouper = self.dataset.groupby('event_parent_flash_id').groups
        count = [len(grouper[eid]) if (eid in grouper) else 0
                 for eid in self.dataset['flash_id'].data]
        flash_child_event_count = xr.DataArray(count, 
                                               dims=[self.fl_dim,])
        self.dataset['flash_child_event_count'] = flash_child_event_count
 
 
    @property
    def fov_bounds(self):
#         lat_bnd = self.dataset.lat_field_of_view_bounds.data
#         lon_bnd = self.dataset.lon_field_of_view_bounds.data
        lat_bnd = self.dataset.event_lat.min().data, self.dataset.event_lat.max().data
        lon_bnd = self.dataset.event_lon.min().data, self.dataset.event_lon.max().data
        return lon_bnd,lat_bnd

        
    def subset_flashes(self, lon_range=None, lat_range=None, 
               x_range=None, y_range=None,
               min_events=None, min_groups=None):
        """ Subset the dataset based on longitude, latitude, the minimum
            number of events per flash, and/or the minimum number of groups 
            per flash.
            
            Applies subsetting only the the flashes, and then retrieves all
            events and groups that go with those flash ids. 
            
            If a flash's centroid is within the bounding box by has events that
            straddle the bounding box edge, all events will still be returned.
            Same goes for groups. Therefore, the group and event locations are
            not strictly within the bounding box.

            If either or both of x_range and y_range are not None, the flash
            flash lon,lat is converted to fixed grid coords and filtered on
            that basis in addition to any other filtering.
        """
        good = np.ones(self.dataset.flash_id.shape[0], dtype=bool)
        flash_data = self.dataset
        
        if (x_range is not None) | (y_range is not None):
            nadir_lon = flash_data.lon_field_of_view.data
            flash_x, flash_y = ltg_ellps_lon_lat_to_fixed_grid(
                flash_data.flash_lon.data, flash_data.flash_lat.data,
                nadir_lon)
        if (x_range is not None):
            good &= ((flash_x < x_range[1]) & (flash_x > x_range[0]))
        if (y_range is not None):
            good &= ((flash_y < y_range[1]) & (flash_y > y_range[0]))
        if lon_range is not None:
            good &= ((flash_data.flash_lon < lon_range[1]) & 
                     (flash_data.flash_lon > lon_range[0])).data
        if lat_range is not None:
            good &= ((flash_data.flash_lat < lat_range[1]) & 
                     (flash_data.flash_lat > lat_range[0])).data
        if min_events is not None:
            good &= (flash_data.flash_child_event_count >= min_events).data
        if min_groups is not None:
            good &= (flash_data.flash_child_group_count >= min_groups).data

        flash_ids = flash_data.flash_id[good].data
        return self.get_flashes(flash_ids)
    
    def get_flashes(self, flash_ids):
        """ Subset the dataset to a some flashes with ids given by a list of
            flash_ids. Can be used to retrieve a single flash by passing a
            single element list.
        """
        these_flashes = self.reduce_to_entities('flash_id', flash_ids)
        return these_flashes
        

