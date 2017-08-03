import numpy as np
import pandas as pd
import xarray as xr

from glmtools.io.glm import GLMDataset
from lmatools.io.LMA_h5_file import LMAh5Collection
from lmatools.lasso.cell_lasso_timeseries import TimeSeriesGenericFlashSubset
from lmatools.lasso.energy_stats import TimeSeriesPolygonLassoFilter 

# from glmtools.io.glm import fix_unsigned
def fix_unsigned(arg):
    return arg

# from lmatools.io.LMA_h5_file import LMAh5Collection

# h5LMAfiles=('/data/20130606/LMA/LYLOUT_130606_032000_0600.dat.flash.h5',)

# h5s = LMAh5Collection(h5LMAfiles, base_date=panels.basedate, min_points=10)    
# for events, flashes in h5s: # get a stream of events, flashes 
#     print events.dtype
#     print flashes.dtype

def read_flashes(glm, target, base_date=None, lon_range=None, lat_range=None,
                 min_events=None, min_groups=None):
    """ This routine is the data pipeline source, responsible for pushing out 
        events and flashes. Using a different flash data source format is a matter of
        replacing this routine to read in the necessary event and flash data.
        
        If target is not None, target is treated as an lmatools coroutine
        data source. Otherwise, return `(events, flashes)`.
     """
    
    if ((lon_range is not None) | (lat_range is not None) |
        (min_events is not None) | (min_groups is not None)):
        # only subset if we have to
        flash_data = glm.subset_flashes(lon_range=lon_range, lat_range=lat_range,
                        min_events=min_events, min_groups=min_groups)
    else:
        flash_data = glm.dataset

    try:
        events, flashes = mimic_lma_dataset(flash_data, base_date)
        if target is not None:
            if events.shape[0] >= 1:
                target.send((events, flashes))
                del events, flashes
        else:
            return events, flashes
    except KeyError as ke:
        err_txt = 'Skipping {0}\n    ... assuming a flash, group, or event with id {1} does not exist'
        print(err_txt.format(glm.dataset.dataset_name, ke))


def sec_since_basedate(t64, basedate):
    """ given a numpy datetime 64 object, and a datetime basedate, 
        return seconds since basedate"""
    
    t_series = pd.Series(t64) - basedate
    t = np.fromiter((dt.total_seconds() for dt in t_series), dtype='float64')
    return t

def _fake_lma_from_glm(flash_data, basedate):
    """ `flash_data` is an xarray dataset of flashes, groups, and events for
         (possibly more than one) lightning flash. `flash_data` can be generated
         with `GLMDataset.subset_flashes` or `GLMDataset.get_flashes`.
    """
    # These are the dtypes in the LMA HDF5 data files
    event_dtype=[('flash_id', '<i4'), 
                 ('alt', '<f4'), 
#                  ('charge', 'i1'), ('chi2', '<f4'), ('mask', 'S4'), ('stations', 'u1'),
                 ('lat', '<f4'), ('lon', '<f4'), ('time', '<f8'),
                 ('power', '<f4'), ]
    flash_dtype=[('area', '<f4'),  ('total_energy', '<f4'), 
                 #('volume', '<f4'), 
                 ('specific_energy', '<f4'), 
                 ('ctr_lat', '<f4'), ('ctr_lon', '<f4'), 
                 ('ctr_alt', '<f4'), 
                 ('start', '<f8'), ('duration', '<f4'), 
                 ('init_lat', '<f4'), ('init_lon', '<f4'), 
                 ('init_alt', '<f4'),# ('init_pts', 'S256'), 
                 ('flash_id', '<i4'),  ('n_points', '<i2'),  ]

    
    flash_np = np.empty_like(flash_data.flash_id.data, dtype=flash_dtype)
    event_np = np.empty_like(flash_data.event_id.data, dtype=event_dtype)

    
    # for each event get the group id
    # for each group id get the flash id

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return event_np, flash_np
        
    event_np['flash_id'] = flash_data.event_parent_flash_id.data
    # Doesn't work for more than one flash in the data table.
    # event_np['flash_id'] = flash_data.event_parent_flash_id.data

#     ev_lats, ev_lons = fix_event_locations(flash_data.event_lat, flash_data.event_lon, is_xarray=True)
    event_np['lat'] = fix_unsigned(flash_data.event_lat) # ev_lats
    event_np['lon'] = fix_unsigned(flash_data.event_lon)
    t_event = sec_since_basedate(flash_data.event_time_offset.data, basedate)
    event_np['time'] = t_event
    event_np['power'] = fix_unsigned(flash_data.event_energy) 

    flash_np['area'] = fix_unsigned(flash_data.flash_area)
    flash_np['total_energy'] = fix_unsigned(flash_data.flash_energy)
    flash_np['ctr_lon'] = flash_data.flash_lon.data
    flash_np['ctr_lat'] = flash_data.flash_lat.data
    flash_np['init_lon'] = flash_data.flash_lon.data
    flash_np['init_lat'] = flash_data.flash_lat.data
    t_start = sec_since_basedate(flash_data.flash_time_offset_of_first_event.data, basedate)
    t_end = sec_since_basedate(flash_data.flash_time_offset_of_last_event.data, basedate)
    flash_np['start'] = t_start
    flash_np['duration'] = t_end-t_start
    flash_np['flash_id'] = flash_data.flash_id.data
    flash_np['n_points'] = flash_data.number_of_events.shape[0]
    
    # Fake the altitude data
    event_np['alt'] = 0.0
    flash_np['ctr_alt'] = 0.0
    flash_np['init_alt'] = 0.0
    
    # Fake the specific energy data
    flash_np['specific_energy'] = 0.0
    
    return event_np, flash_np



def mimic_lma_dataset(flash_data, basedate):
    """ Mimic the LMA data structure from GLM """        
    events, flashes = _fake_lma_from_glm(flash_data, basedate)
    return events, flashes



class GLMncCollection(LMAh5Collection):
    """ Mimic the the events, flashes and time selection behavior of 
        LMAh5Collection but with GLM data files instead. 
        kwarg min_points is used for the minimum number of events and
        min_groups is used for the minimum number of groups.
    """
    def __init__(self, *args, **kwargs):
        self.min_groups = kwargs.pop('min_groups', None)
        self.lat_range = kwargs.pop('lat_range', None)
        self.lon_range = kwargs.pop('lon_range', None)
        super().__init__(*args, **kwargs)

    def _table_times_for_file(self, fname):
        """ Called once by init to set up frame lookup tables and yield 
            the frame start times. _time_lookup goes from 
            datetime->(h5 filename, table_name).
        """
        glm = GLMDataset(fname, calculate_parent_child=False)
        # Get the time, using 'seconds' resolution because GLM files are
        # produced on 20 s boundaries (or any number of even seconds)
        t_start = glm.dataset.product_time.data.astype('M8[s]').astype('O')
        # In the LMA API, we track the table name in the H5 file that goes
        # with the time. No such thing exists for GLM data.
        self._time_lookup[t_start] = (fname, None)
        yield t_start

    def data_for_time(self, t0):
        """ Return events, flashes whose start time matches datetime t0.
        
        events['time'] and flashes['start'] are corrected if necessary
        to be referenced with respect to self.base_date.
        """
        fname, table_name = self._time_lookup[t0]
        glm = GLMDataset(fname)
        events, flashes = read_flashes(glm, None, base_date=self.base_date,
                                       min_events=self.min_points,
                                       min_groups=self.min_groups,
                                       lon_range=self.lon_range, 
                                       lat_range=self.lat_range)
        print('data from {0}'.format(fname))
        return events, flashes
        
class TimeSeriesGLMFlashSubset(TimeSeriesGenericFlashSubset):
    def __init__(self, glm_filenames, t_start, t_end, dt, base_date=None, 
                    min_events=None, min_groups=None, 
                    lon_range=None, lat_range=None):
        super(TimeSeriesGLMFlashSubset, self).__init__(t_start, t_end, dt, 
                                                    base_date=None)
        self.lma = GLMncCollection(glm_filenames, base_date=self.base_date,
                                   min_points=min_events, min_groups=min_groups,
                                   lat_range=lat_range, lon_range=lon_range)
                                   
class TimeSeriesGLMPolygonFlashSubset(TimeSeriesGLMFlashSubset):
    # This duplicates all the code from 
    # lmatools.lasso.cell_lasso_timeseries.TimeSeriesPolygonFlashSubset
    # The only change is the superclass and class name.
    # There's surely a way to do this more elegantly.
    def __init__(self, *args, **kwargs):
        # could also accept coord_names and time_key kwargs here, but those
        # should be standardized in the lmatools h5 format, so we hard code
        # them below
        self.polys = kwargs.pop('polys', [])
        self.t_edges_polys = kwargs.pop('t_edges_polys', [])

        super(TimeSeriesGLMPolygonFlashSubset, self).__init__(*args, **kwargs)
        
        # strictly speaking, we don't even need the time series part; that's been done
        # and we're just lassoin' the points. 
        # But, this code is known to work, so we just reuse it here.
        self.fl_lassos = TimeSeriesPolygonLassoFilter(coord_names=('init_lon', 'init_lat'), time_key='start',
                                         time_edges=self.t_edges_polys, polys=self.polys, basedate=self.base_date )
        self.ev_lassos = TimeSeriesPolygonLassoFilter(coord_names=('lon', 'lat'), time_key='time',
                                         time_edges=self.t_edges_polys, polys=self.polys, basedate=self.base_date)
        self.grid_lassos = TimeSeriesPolygonLassoFilter(coord_names=('lon', 'lat'), time_key='t',
                                         time_edges=self.t_edges_polys, polys=self.polys, basedate=self.base_date)
    
    def gen_chopped_events_flashes(self, *args, **kwargs):
        parent = super(TimeSeriesGLMPolygonFlashSubset, self)
        for ch_ev_series, ch_fl_series in parent.gen_chopped_events_flashes(*args, **kwargs):
            # This gives a time series for each HDF5 LMA file. Next loop
            # over each chopped time series window.
            # Apply polygon filter to time series created by superclass, yield chopped events and flashes
            lassoed_ev = [ch_ev[self.ev_lassos.filter_mask(ch_ev)] for ch_ev in ch_ev_series]
            lassoed_fl = [ch_fl[self.fl_lassos.filter_mask(ch_fl)] for ch_fl in ch_fl_series]
            yield lassoed_ev, lassoed_fl
            
