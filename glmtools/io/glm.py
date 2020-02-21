import itertools
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

from glmtools.io.traversal import OneToManyTraversal
from glmtools.io.lightning_ellipse import ltg_ellps_lon_lat_to_fixed_grid
from glmtools.io.lightning_ellipse import ltg_ellpse_rev

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

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
    def __init__(self, filename, calculate_parent_child=True, ellipse_rev=-1,
                 check_area_units=True, change_energy_units=True,
                 fix_bad_DO07_times=True):
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

            ellipse_rev: Lightning ellipsoid associated with lat/lon values
                    in filename.
                -1 (default) : infer ellipsoid from GLM filename
                0 : version at launch
                1 : first revision, lowering equatorial height to 14 km.

            check_area_units: If True (default) check the units on flash
                and group area and convert to km^2 if in m^2.
            change_energy_units: If True (default) change the units of flash,
                group, and event energy to nJ.
            fix_bad_DO07_times: If True (default), correct for the missing
                _Unsigned attribute for the ~month in Oct-Nov 2018 when the
                problem was present.
        """
        self.entity_ids = ['flash_id', 'group_id', 'event_id']
        self.parent_ids = ['group_parent_flash_id', 'event_parent_group_id']

        if isinstance(filename, xr.Dataset):
            dataset = filename
            super().__init__(dataset,
                             self.entity_ids, self.parent_ids)
        else:
            dataset = xr.open_dataset(filename)
        self._filename = filename
        self.ellipse_rev = ellipse_rev

        self.fov_dim = 'number_of_field_of_view_bounds'
        self.wave_dim = 'number_of_wavelength_bounds'
        self.time_dim = 'number_of_time_bounds'
        self.gr_dim = 'number_of_groups'
        self.ev_dim = 'number_of_events'
        self.fl_dim = 'number_of_flashes'

        if calculate_parent_child:
            # sets self.dataset
            super().__init__(dataset,
                             self.entity_ids, self.parent_ids)
            self.__init_parent_child_data()
            self.__init_fixed_grid_data()
            # self.__init_event_lut()
        else:
            self.dataset = dataset

        if fix_bad_DO07_times:
            did_fix = self._check_and_fix_missing_unsigned_time(filename)
        if check_area_units:
            did_fix = self._check_area_units()
        if change_energy_units:
            did_fix = self._change_energy_units()

    def __init_parent_child_data(self):
        """ Calculate implied parameters that are useful for analyses
            of GLM data.
        """
        if ((self.dataset.dims['number_of_flashes'] == 0) |
            (self.dataset.dims['number_of_groups'] == 0) |
            (self.dataset.dims['number_of_events'] == 0)):
            no_data = True
            log.warning('File {0} has no data, skipping it'.format(
                        self._filename))

        else:
            no_data = False

        if no_data:
            # At least one of the dimensiosn of the dataset is empty, so the
            # the traversal of the flash/group/event hierarchy cannot take
            #place. xarray's groupby raises a ValueError on empty DataArrays,
            # and groupby is used in the traversal.
            flash_ids = []
        else:
            flash_ids = self.replicate_parent_ids('flash_id',
                                                  'event_parent_group_id'
                                                  )
        event_parent_flash_id = xr.DataArray(flash_ids, dims=[self.ev_dim,])
        self.dataset['event_parent_flash_id'] = event_parent_flash_id

        if no_data:
            flash_child_count = []
            group_child_count = []
            count = []
        else:
            all_counts = self.count_children('flash_id', 'event_id')
            flash_child_count = all_counts[0]
            group_child_count = all_counts[1]
            # we can use event_parent_flash_id to get the flash_child_event_count
            # need a new groupby on event_parent_flash_id
            # then count number of flash_ids that match in the groupby
            # probably would be a good idea to add this to the traversal class
            grouper = self.dataset.groupby('event_parent_flash_id').groups
            count = [len(grouper[eid]) if (eid in grouper) else 0
                     for eid in self.dataset['flash_id'].data]

        flash_child_group_count = xr.DataArray(flash_child_count,
                                               dims=[self.fl_dim,])
        self.dataset['flash_child_group_count'] = flash_child_group_count
        group_child_event_count = xr.DataArray(group_child_count,
                                               dims=[self.gr_dim,])
        self.dataset['group_child_event_count'] = group_child_event_count
        flash_child_event_count = xr.DataArray(count,
                                               dims=[self.fl_dim,])
        self.dataset['flash_child_event_count'] = flash_child_event_count

    def _change_energy_units(self):
        """ Change the flash energy units to nJ.
        Doesn't change the scale/offset, so the discretization of values
        if this L2 dataset were to be written to disk. glmtools does not
        do this, but it might be noticed if someone tried to!
        """
        changed_flash_energy, changed_group_energy, changed_event_energy = (
            False, False, False)
        if self.dataset.flash_energy.units == 'J':
            self.dataset['flash_energy'] = self.dataset['flash_energy']*1.0e9
            self.dataset.flash_energy.attrs['units'] = 'nJ'
            changed_flash_energy = True
        else:
            raise ValueError("Flash energy units have changed from PUG v.2.0")
        if self.dataset.group_energy.units == 'J':
            self.dataset['group_energy'] = self.dataset['group_energy']*1.0e9
            self.dataset.group_energy.attrs['units'] = 'nJ'
            changed_group_energy = True
        else:
            raise ValueError("Group energy units have changed from PUG v.2.0")
        if self.dataset.event_energy.units == 'J':
            self.dataset['event_energy'] = self.dataset['event_energy']*1.0e9
            self.dataset.event_energy.attrs['units'] = 'nJ'
            changed_event_energy = True
        else:
            raise ValueError("Event energy units have changed from PUG v.2.0")
        return changed_flash_energy, changed_group_energy, changed_event_energy

    def _check_area_units(self):
        fixed_flash_area, fixed_group_area = False, False
        if self.dataset.flash_area.units == 'm2':
            self.dataset['flash_area'] = self.dataset['flash_area']/1.0e6
            self.dataset.flash_area.attrs['units'] = 'km2'
            fixed_flash_area = True
            log.debug('New flash area units: {0}'.format(
                        self.dataset.flash_area.units))
        if self.dataset.group_area.units == 'm2':
            self.dataset['group_area'] = self.dataset['group_area']/1.0e6
            self.dataset.group_area.attrs['units'] = 'km2'
            fixed_group_area = True
            log.debug('New group area units: {0}'.format(
                        self.dataset.group_area.units))
        return fixed_flash_area, fixed_group_area

    def _check_event_xy(self, x_range=(-0.1349, 0.1349),
                              y_range=(-0.1349, 0.1349)):
        """ Return the flash IDs corresponding to those events whose fixed grid
            coordinates are within the expected range. The default values for
            x_range and y_range are the maximum range covered by the GLM corner
            point lookup table. If values larger than this are passed to this
            function they will be clipped to these values.
        """
        x_valid = y_valid = (-0.1349, 0.1349)
        x_range = (max((x_valid[0], x_range[0])),
                   min((x_valid[1], x_range[1])))
        y_range = (max((y_valid[0], y_range[0])),
                   min((y_valid[1], y_range[1])))
        log.debug("Subsetting bad events with final ranges {0}".format(
            (x_range, y_range)))
        good = np.ones(self.dataset.event_id.shape[0], dtype=bool)
        event_x = self.dataset.event_x.data
        good &= ((event_x < x_range[1]) & (event_x > x_range[0]))
        event_y = self.dataset.event_y.data
        good &= ((event_y < y_range[1]) & (event_y > y_range[0]))
        flash_ids = self.dataset.event_parent_flash_id.data[good]
        bad_ids = np.unique(self.dataset.event_parent_flash_id.data[~good])
        log.debug('Flash IDs with bad event x,y are {0}'.format(bad_ids))
        # for bad_id in bad_ids:
        #     log.debug("{0}".format(self.get_flashes([bad_id])))
        return np.unique(flash_ids)

    def _check_and_fix_missing_unsigned_time(self, filename):
        """ Check for the missing _Unsigned attribute on files created as part of the
        D0.07 build of the operational environment. Correct if present. The problem
        was only present for less than a month, and this function does nothing if
        the data file is outside that time range.

        Modifies self.dataset to have the correct times.
        """
        vars_to_correct = ['event_time_offset','group_time_offset',
                           'group_frame_time_offset',
                           'flash_time_offset_of_first_event',
                           'flash_time_offset_of_last_event',
                           'flash_frame_time_offset_of_first_event',
                           'flash_frame_time_offset_of_last_event']

        start_g16_problem_date = np.datetime64('2018-10-15T00:00:00')
        end_g16_problem_date = np.datetime64('2018-11-06T00:00:00')
        prod_tmin, prod_tmax = self.dataset.product_time_bounds.data
        in_time = (prod_tmin >= start_g16_problem_date) & (prod_tmax <= end_g16_problem_date)
        # The date range is approximate, and corresponds to when D0.07 first was
        # in production in the OE, and before the _Unsigned attribute was added to
        # the production data files. We don't know the exact start time, but fortunately
        # the group_frame_time_offset was also added to D0.07, so we look for that attribute
        # within the date range. Later, adding the attribute to a file that already has it
        # is no problem, so we don't worry about the last time on Nov 6 when the fix is needed.
        if (in_time & hasattr(self.dataset,'group_frame_time_offset')):
            unmod = xr.open_dataset(filename, mask_and_scale=False, decode_cf=False)
            time_dataset = xr.Dataset()
            for var in vars_to_correct:
                # Add the _Unsigned attribute
                da = getattr(unmod,var)
                da.attrs['_Unsigned']='true'
                time_dataset[var] = da

            decoded = xr.decode_cf(time_dataset)

            # Copy corrected time variables over to the new dataset
            for var in vars_to_correct:
                self.dataset[var] = decoded[var]
            unmod.close()
            return True
        else:
            return False

    @property
    def fov_bounds(self):
#         lat_bnd = self.dataset.lat_field_of_view_bounds.data
#         lon_bnd = self.dataset.lon_field_of_view_bounds.data
        lat_bnd = self.dataset.event_lat.min().data, self.dataset.event_lat.max().data
        lon_bnd = self.dataset.event_lon.min().data, self.dataset.event_lon.max().data
        return lon_bnd,lat_bnd


    def subset_flashes(self, lon_range=None, lat_range=None,
               x_range=None, y_range=None, check_event_xy=True,
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

            If check_event_xy is True (default), filter flashes to ensure
            event_x and event_y are within the expected field of view. If
            provided, x_range and y_range are used, but default to and are never
            wider than +/- 0.1349 radians.
        """
        good = np.ones(self.dataset.flash_id.shape[0], dtype=bool)
        flash_data = self.dataset

        if (x_range is not None):
            flash_x = self.dataset.flash_x.data
            good &= ((flash_x < x_range[1]) & (flash_x > x_range[0]))
        if (y_range is not None):
            flash_y = self.dataset.flash_y.data
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

        if check_event_xy:
            check_event_kw = {}
            if x_range is not None: check_event_kw['x_range'] = x_range
            if y_range is not None: check_event_kw['y_range'] = y_range
            # log.debug("Subsetting bad events with override ranges {0}".format(
                # check_event_kw))
            good_event_flash_ids = self._check_event_xy(**check_event_kw)
        flash_ids = list(set(flash_ids) and set(good_event_flash_ids))
        return self.get_flashes(flash_ids)

    def get_flashes(self, flash_ids):
        """ Subset the dataset to a some flashes with ids given by a list of
            flash_ids. Can be used to retrieve a single flash by passing a
            single element list.
        """
        these_flashes = self.reduce_to_entities('flash_id', flash_ids)
        # lutevents = get_lutevents(these_flashes)
        # print(lutevents)
        # these_flashes.update(lutevents)
        return these_flashes

    def __init_fixed_grid_data(self):
        """ Calculate the fixed grid coordinates for flashes, groups, and events
            and save them to self.dataset as new variables named
            event_x, event_y, group_x, group_y, flash_x, flash_y
        """
        nadir_lon = self.dataset.lon_field_of_view.data
        ellipse_rev = self.ellipse_rev
        if ellipse_rev < 0:
            log.info("Inferring lightning ellipsoid from GLM product time")
            pt = self.dataset.product_time.dt
            date = datetime(pt.year, pt.month, pt.day,
                            pt.hour, pt.minute, pt.second)
            ellipse_rev = ltg_ellpse_rev(date)
        log.info("Using lightning ellipsoid rev {0}".format(ellipse_rev))

        event_x, event_y = ltg_ellps_lon_lat_to_fixed_grid(
            self.dataset.event_lon.data, self.dataset.event_lat.data,
            nadir_lon, ellipse_rev)
        self.dataset['event_x'] = xr.DataArray(event_x, dims=[self.ev_dim,])
        self.dataset['event_y'] = xr.DataArray(event_y, dims=[self.ev_dim,])

        group_x, group_y = ltg_ellps_lon_lat_to_fixed_grid(
            self.dataset.group_lon.data, self.dataset.group_lat.data,
            nadir_lon, ellipse_rev)
        self.dataset['group_x'] = xr.DataArray(group_x, dims=[self.gr_dim,])
        self.dataset['group_y'] = xr.DataArray(group_y, dims=[self.gr_dim,])

        flash_x, flash_y = ltg_ellps_lon_lat_to_fixed_grid(
            self.dataset.flash_lon.data, self.dataset.flash_lat.data,
            nadir_lon, ellipse_rev)
        self.dataset['flash_x'] = xr.DataArray(flash_x, dims=[self.fl_dim,])
        self.dataset['flash_y'] = xr.DataArray(flash_y, dims=[self.fl_dim,])

    # def __init_event_lut(self):
    #     """ Used on init to set up the event lookup table for the whole dataset """
    #     lutevents = get_lutevents(self.dataset)
    #     self.dataset.update(lutevents)


def get_lutevents(dataset, scale_factor=28e-6, event_dim='number_of_events',
        x_range=(-0.31, 0.31), y_range=(-0.31, 0.31)):
    """ Build an event lookup table. Assign each event location a "sort of"
        pixel ID based on its fixed grid coordinates, discretized to some step
        interval that is less than the minimum pixel spacing of 224 microrad=8
        km at nadir.

        A new location is assigned to each discretized location (mean of the
        locations of  the constituent events). The time is assigned, uniformly,
        to be the dataset's product time attribute.

        The event lookup table is accompanied by pre-accumulated data at each
        discretized location: the flash, group and event counts; total flash and
        group areas; total event energy.

        Returns a new dataset with dimension "lutevent_id", having an index of
        the same name. The dataset is a (shallow) copy, but a new xarray object.

        If needed, returned dataset lutevents can be added to the original
        dataset with dataset.update(lutevents).

        If the pixel ID were stored as a 32 bit unsigned integer,
        (0 to 4294967295) that is 65536 unique values for a square (x,y) grid,
        the minimum safe scale factor for the span of the full disk is
        (0.62e6 microradians)/65536 = 9.46 microradians
        which is a bit large. Therefore, the implementation uses 64 bit unsigned
        integers to be safe.

        Arguments:
        dataset: GLM dataset in xarray format

        Keyword arguments:
        scale_factor: discretization interval, radians (default 28e-6)
        x_range, y_range: range of possible fixed grid coordinate values
            (default -/+.31 radians, which is larger than the
            full disk at geo. Ref: GOES-R PUG Vol. 3, L1b data.)
    """
    # Make a copy of the dataset so we can update it and return a copy.
    # xarray copys are shallow/cheap, and the xarray docs promote returning new
    # datasets http://xarray.pydata.org/en/stable/combining.html
    dataset = dataset.copy()
    event_x, event_y = dataset.event_x.data, dataset.event_y.data
    event_energy = dataset.event_energy.data
    product_time = dataset.product_time.data
    ev_flash_id = dataset.event_parent_flash_id.data
    ev_group_id = dataset.event_parent_group_id.data
    flash_area = dataset.flash_area.data
    group_area = dataset.group_area.data

    xy_id = discretize_2d_location(event_x, event_y, scale_factor, x_range, y_range)
    dataset['event_parent_lutevent_id'] = xr.DataArray(xy_id, dims=[event_dim,])
    eventlut_groups = dataset.groupby('event_parent_lutevent_id')
    flash_id_groupby = dataset.groupby('flash_id')
    group_id_groupby = dataset.groupby('group_id')
    n_lutevents = len(eventlut_groups.groups)

    # Create a new dimension for the reduced set of events, with their
    # properties aggregated.
    # - Sum: event_energy, flash_area, group_area
    # - Mean: event_x, event_y
    # - Count: event_id; unique flash_id, group_id
    # - Min: flash_area
    eventlut_dtype = [('lutevent_id', 'u8'),
                      ('lutevent_x', 'f8'),
                      ('lutevent_y', 'f8'),
                      ('lutevent_energy','f8'),
                      ('lutevent_count', 'f4'),
                      ('lutevent_flash_count', 'f4'),
                      ('lutevent_group_count', 'f4'),
                      ('lutevent_total_flash_area', 'f8'),
                      ('lutevent_total_group_area', 'f8'),
                      ('lutevent_time_offset', '<M8[ns]'),
                      ('lutevent_min_flash_area', 'f8')
                      ]

    lut_iter = event_lut_iter(eventlut_groups, flash_id_groupby, group_id_groupby,
                   event_x, event_y, event_energy, product_time,
                   ev_flash_id, ev_group_id, flash_area, group_area)
    event_lut = np.fromiter(lut_iter, dtype=eventlut_dtype, count=n_lutevents)
    lutevents = xr.Dataset.from_dataframe(
                    pd.DataFrame(event_lut).set_index('lutevent_id'))
    dataset.update(lutevents)
    return dataset


def event_lut_iter(event_lut_groupby, flash_groupby, group_groupby,
                   event_x, event_y, event_energy, product_time,
                   ev_flash_id, ev_group_id, flash_area, group_area):
    flash_groups = flash_groupby.groups
    group_groups = group_groupby.groups
    total_abs_area_delta = 0.0
    total_area = 0.0
    for xy_id, evids in event_lut_groupby.groups.items():
        flash_ids = np.unique(ev_flash_id[evids])
        group_ids = np.unique(ev_group_id[evids])
        flash_count, group_count = len(flash_ids), len(group_ids)
        # old_flash_area = sum((flash_area[flash_groups[fid]].sum()
        #     for fid in flash_ids))
        replicated_flashes = list(itertools.chain.from_iterable(
            flash_groups[fid] for fid in flash_ids))
        total_flash_area = flash_area[replicated_flashes].sum()
        # total_abs_area_delta += np.abs(total_flash_area - old_flash_area)
        # total_area += total_flash_area
        # print(total_abs_area_delta, total_area)
        # old_group_area = sum((group_area[group_groups[gid]].sum()
        #     for gid in group_ids))
        replicated_groups = list(itertools.chain.from_iterable(
            group_groups[gid] for gid in group_ids))
        total_group_area = group_area[replicated_groups].sum()
        # total_abs_area_delta += np.abs(total_group_area - old_group_area)
        # total_area += total_group_area
        # print(total_abs_area_delta, total_area)
        min_flash_area = min((flash_area[flash_groups[fid]].min()
            for fid in flash_ids))
        yield (xy_id,
               event_x[evids].mean(),
               event_y[evids].mean(),
               event_energy[evids].sum(),
               len(evids),
               flash_count,
               group_count,
               total_flash_area,
               total_group_area,
               product_time,
               min_flash_area
               )


def discretize_2d_location(x, y, scale, x_range, y_range, int_type='uint64'):
    """ Calculate a unique location ID for a 2D position given some
        discretization interval

        Arguments:
        x, y: coordinates, float arrays
        scale: discretization interval
        x_range, y_range: 2-tuple giving min and max x and y values

        Keyword arguments:
        int_type: numpy dtype of xy_id. 64 bit unsigned int by default, since 32 bit
            is limited to a 65536 pixel square grid.

        Returns:
        xy_id = unique pixel ID, 64 bit unsigned integer
    """
    x_offset = x_range[0]
    y_offset = y_range[0]
    discr_x_max = np.array((x_range[1]-x_offset)/scale, dtype=int_type)
    discr_y_max = np.array((y_range[1]-y_offset)/scale, dtype=int_type)
    x_discr = ((x - x_offset) / scale).astype(int_type)
    y_discr = ((y - y_offset) / scale).astype(int_type)
    xy_discr = x_discr + y_discr*discr_x_max
    return xy_discr
