#!/usr/bin/env python
import argparse
parse_desc = """Calculate flash statistics within a (possibly time-evolving) 
polygon, optionally filtering on a variety of flash parameters."""

lasso_log_help = """ path_to_lasso_log.txt is a path to a log file of cell
lassos in json format, as created by the grid analysis GUI notebook that
operates on the NetCDF flash files."""

path_to_sort_help = """path_to_sort_results is a path to a standard directory
of flash sorting results produced by lmatools. glmtools will look within this 
directory for a grid_files directory that has GLM gridded data on the usual 
yyy/mon/dd/*.nc path.
"""

outdir_help = """outdir is created in within the figures-length folder
inside path_to_sort_results. This way multiple runs with different lassos
on the same dataset are stored together. figures-length is created if it does
not exist. """

parser = argparse.ArgumentParser(description=parse_desc)
parser.add_argument(dest='filenames',metavar='OR_GLM-L2-LCFA*.nc', nargs='*')
parser.add_argument('-s', dest='path_to_sort_results', action='store',
                    help=path_to_sort_help,)
parser.add_argument('--stdpath', dest='use_standard_path', action='store_true',
                    help='Look for data in h5_files directory in path_to_sort_results')                    
parser.add_argument('-o', '--output_dir', metavar='directory', required=True,
                    dest='outdir', action='store', help=outdir_help)
parser.add_argument('-l', '--lasso', metavar='filename', required=True,
                    dest='lasso_log', action='store',
                    help=lasso_log_help)
parser.add_argument('--minalt', metavar='float km',
                    type=float, dest='min_alt', action='store', default=-1.0)
parser.add_argument('--maxalt', metavar='float km',
                    type=float, dest='max_alt', action='store', default=100.0)
parser.add_argument('--minenergy', metavar='float',
                    type=float, dest='min_energy', action='store', default=0.0)
parser.add_argument('--maxenergy', metavar='float',
                    type=float, dest='max_energy', action='store', default=1.0e38)
parser.add_argument('-a', '--minarea', metavar='float km^2',
                    type=float, dest='min_area', action='store', default=0.0)
parser.add_argument('-A', '--maxarea', metavar='float km^2',
                    type=float, dest='max_area', action='store', default=1.0e10)
parser.add_argument('--nevents', metavar='minimum events per flash', type=int,
                    dest='min_events', action='store', default=1,
                    help='minimum number of events per flash')
parser.add_argument('--ngroups', metavar='minimum groups per flash', type=int,
                    dest='min_groups', action='store', default=1,
                    help='minimum number of groups per flash')
parser.add_argument('--skip3d', dest='do_3d', action='store_false',
                    help='Skip calculations requiring 3D flash position data')
parser.add_argument('--skipspectra', dest='do_energy_spectra', 
                    action='store_false', help='Skip plots of energy spectra')

args = parser.parse_args()

min_area = args.min_area
max_area = args.max_area
min_alt = args.min_alt * 1000.0
max_alt = args.max_alt * 1000.0
min_energy = args.min_energy
max_energy = args.max_energy

min_events = args.min_events
if min_events < 2:
    min_events = None
min_groups = args.min_groups
if min_groups < 2:
    min_groups = None

do_3d = args.do_3d
do_energy_spectra = args.do_energy_spectra
use_standard_path = args.use_standard_path

polylog = args.lasso_log
path_to_sort_results = args.path_to_sort_results
outdir_name_from_user = args.outdir
input_filenames = args.filenames

##### END PARSING #####

import os, sys, errno
from datetime import datetime, timedelta

import numpy as np
from numpy.lib.recfunctions import append_fields, stack_arrays
import matplotlib.pyplot as plt

from lmatools.grid.grid_collection import LMAgridFileCollection
from lmatools.flash_stats import plot_energy_from_area_histogram, get_energy_spectrum_bins,  bin_center, plot_energies

from lmatools.lasso.energy_stats import flash_size_stats, plot_flash_stat_time_series, plot_energy_stats, plot_tot_energy_stats
from lmatools.lasso.length_stats import FractalLengthProfileCalculator
from lmatools.lasso.cell_lasso_util import read_poly_log_file, polys_to_bounding_box, h5_files_from_standard_path, nc_files_from_standard_path
from lmatools.lasso.cell_lasso_timeseries import TimeSeriesPolygonFlashSubset

from lmatools.lasso import empirical_charge_density as cd ###NEW

from glmtools.io.mimic_lma import TimeSeriesGLMPolygonFlashSubset

# =====
# Read polygon data and configure output directories
# =====
polys, t_edges_polys = read_poly_log_file(polylog)

all_poly_lons =  np.asarray(polys)[:,:,0]
all_poly_lats =  np.asarray(polys)[:,:,1]
lonmin, lonmax = all_poly_lons.min(), all_poly_lons.max() 
latmin, latmax = all_poly_lats.min(), all_poly_lats.max() 

t_start, t_end = min(t_edges_polys), max(t_edges_polys)
dt = timedelta(minutes=1)

outdir=os.path.join(path_to_sort_results, 'figures-length/', outdir_name_from_user)
try:
    os.makedirs(outdir)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(outdir):
        pass

# =====
# Load data from HDF5 files into a time series of events and flashes
# =====
if use_standard_path:
    h5_filenames = h5_files_from_standard_path(path_to_sort_results, t_start, t_end)
else:
    h5_filenames = input_filenames
flashes_in_poly = TimeSeriesGLMPolygonFlashSubset(h5_filenames, 
                        t_start, t_end, dt, 
                        min_events=min_events, min_groups=min_groups,
                        lon_range=(lonmin, lonmax),
                        lat_range=(latmin, latmax),
                        polys=polys, t_edges_polys=t_edges_polys)
events_series, flashes_series = flashes_in_poly.get_event_flash_time_series()

#-----
find_number_events = []
for events in events_series:
    find_number_events.append(events.shape[0])
number_events = np.asarray(find_number_events).shape[0]
#-----

# filter out flashes based on non-space-time criteria
# filter out events not part of flashes - NOT IMPLEMENTED
def gen_filtered_time_series(events_series, flashes_series, bounds):
    for events, flashes in zip(events_series, flashes_series):
        ev_good = np.ones(events.shape, dtype=bool)
        fl_good = np.ones(flashes.shape, dtype=bool)
        
        for k, (v_min, v_max) in bounds.items():
            if k in events.dtype.names:
                ev_good &= (events[k] >= v_min) & (events[k] <= v_max)
            if k in flashes.dtype.names:
                fl_good &= (flashes[k] >= v_min) & (flashes[k] <= v_max)
        events_filt, flashes_filt = (events[ev_good], flashes[fl_good])
        
        yield (events_filt, flashes_filt)

# used to remove flashes not meeting certain critera
bounds={'area':(min_area, max_area),
        'ctr_alt':(min_alt,max_alt),
        'total_energy':(min_energy, max_energy)
}
filtered_time_series = gen_filtered_time_series(events_series, flashes_series, bounds)
events_series, flashes_series = zip(*filtered_time_series)

# =====
# plot the raw flash locations
# =====
class FlashCentroidPlotter(object):
    def __init__(self, t_sec_min, t_sec_max):
        self.t_sec_min, self.t_sec_max = t_sec_min, t_sec_max
        self.fig = plt.figure(figsize=(11,11))
        self.evax = self.fig.add_subplot(111)
        self.evax.set_ylabel('Latitude (deg)')
        self.evax.set_xlabel('Longitude (deg)')
        self.evax.set_title('Flash event-weighted centroid, \n{0} to {1} seconds'.format(
            t_sec_min, t_sec_max
            # date_min.strftime('%Y-%m-%d %H%M:%S'),
            # date_max.strftime('%Y-%m-%d %H%M:%S')
        ))
        self.colorbar = None
        self.scatters = []

    def plot(self, flashes):
        scart = self.evax.scatter(flashes['init_lon'], flashes['init_lat'],
                      c=flashes['start'], s=4, cmap='viridis',
                      vmin=self.t_sec_min, vmax=self.t_sec_max, edgecolor='none')
        if self.colorbar is None:
            self.colorbar = plt.colorbar(scart, ax=self.evax)
        self.scatters.append(scart)

flash_location_plotter = FlashCentroidPlotter(min(flashes_in_poly.t_edges_seconds),
                                              max(flashes_in_poly.t_edges_seconds))
for flashes in flashes_series:
    flash_location_plotter.plot(flashes)
flash_ctr_filename = 'flash_ctr_{0}_{1}.png'.format(t_start.strftime('%y%m%d%H%M%S'),
                                                    t_end.strftime('%y%m%d%H%M%S'))
flash_location_plotter.fig.savefig(os.path.join(outdir, flash_ctr_filename))


# =======================================
# Loop over each window in the time series and calculate some
# aggregate flash statistical properties
# =======================================

# Set up fractal length calcuations and channel height profiles
if do_3d:
    D = 1.5
    b_s = 200.0
    max_alt, d_alt = 20.0, 0.5
    alt_bins = np.arange(0.0,max_alt+d_alt, d_alt)
    length_profiler = FractalLengthProfileCalculator(D, b_s, alt_bins)
else:
    length_profiler = None

def write_flash_volume(outfile, field):
    import xray as xr
    import pandas as pd
    dat = pd.DataFrame(field).T
    dat.to_csv(outfile)

def gen_flash_summary_time_series(events_series, flashes_series, length_profiler):
    for events, flashes in zip(events_series, flashes_series):
        # reduce all flashes in this time interval to representative moments
        size_stats = flash_size_stats(flashes)
        # for each flash, get a dictionary with 2D and 3D fractal lengths.
        # also includes the volume and point triangulation data.
        if length_profiler is not None:
            per_flash_data, IC_profile, CG_profile = length_profiler.process_events_flashes(events, flashes)
        else:
            IC_profile, CG_profile = None, None
        yield size_stats, IC_profile, CG_profile


time_series = gen_flash_summary_time_series(events_series, flashes_series,
                                            length_profiler)
size_stats, IC_profiles, CG_profiles = zip(*time_series)

size_stats = stack_arrays(size_stats)
iso_start, iso_end = flashes_in_poly.t_edges_to_isoformat(as_start_end=True)
# assume here that iso_start and iso_end are identical-length strings,
# as they should be if the iso format is worth anything.
size_stats = append_fields(size_stats, ('start_isoformat','end_isoformat'),
                                         data=(iso_start, iso_end), usemask=False)                                      
# =====
# Write flash size stats data (see harvest_flash_timeseries) and plot moment time series
# =====
def write_size_stat_data(outfile, size_stats):
    stat_keys = ('start_isoformat','end_isoformat', 'number', 'mean', 'variance',
                 'skewness', 'kurtosis', 'energy', 'energy_per_flash','total_energy','specific_energy')
    header = "# start_isoformat, end_isoformat, number, mean, variance, skewness, kurtosis, energy, energy_per_flash, total_energy, specific_energy\n"
    line_template = "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}\n"
    f = open(outfile, 'w')
    #for start_t, end_t, stats in zip(starts, ends, size_stats):
    f.write(header)
    for stats in size_stats:
        stat_vals = [stats[k] for k in stat_keys]
        line = line_template.format(*stat_vals)
        f.write(line)
    f.close()

stats_filename = os.path.join(outdir,'flash_stats.csv')
stats_figure = os.path.join(outdir,'flash_stats_{0}_{1}.pdf'.format(t_start.strftime('%y%m%d%H%M%S'),
                                                    t_end.strftime('%y%m%d%H%M%S')))
write_size_stat_data(stats_filename, size_stats)
fig = plot_flash_stat_time_series(flashes_in_poly.base_date, flashes_in_poly.t_edges, size_stats)
fig.savefig(stats_figure)

energy_fig = plot_energy_stats(size_stats['specific_energy'], flashes_in_poly.base_date, flashes_in_poly.t_edges, outdir)
energy_figure_name = os.path.join(outdir,'specific_energy.pdf')
energy_fig.savefig(energy_figure_name)

tot_energy_fig = plot_tot_energy_stats(size_stats['total_energy'], flashes_in_poly.base_date, flashes_in_poly.t_edges, outdir)
tot_energy_figure_name = os.path.join(outdir,'total_energy.pdf')
tot_energy_fig.savefig(tot_energy_figure_name)

# =====
# Write profile data to file and plot profile time series
# =====
if do_3d:
    durations_min = (flashes_in_poly.t_edges_seconds[1:] - flashes_in_poly.t_edges_seconds[:-1])/60.0 #minutes
    fig_kwargs = dict(label_t_every=1800.)

    IC_norm_profiles = length_profiler.normalize_profiles(IC_profiles, durations_min)
    CG_norm_profiles = length_profiler.normalize_profiles(CG_profiles, durations_min)

    outfile_base = os.path.join(outdir,
                    'D-{0:3.1f}_b-{1:4.2f}_length-profiles'.format(D,b_s)
                    )

    IC_fig = length_profiler.make_time_series_plot(flashes_in_poly.base_date,
                flashes_in_poly.t_edges_seconds,
                *IC_norm_profiles, **fig_kwargs)
    CG_fig = length_profiler.make_time_series_plot(flashes_in_poly.base_date,
                flashes_in_poly.t_edges_seconds,
                *CG_norm_profiles, **fig_kwargs)
    if 'CG' in flashes_series[0].dtype.names:
        length_profiler.write_profile_data(flashes_in_poly.base_date, flashes_in_poly.t_edges_seconds,
                    outfile_base, *IC_norm_profiles, partition_kind='IC')
        length_profiler.write_profile_data(flashes_in_poly.base_date, flashes_in_poly.t_edges_seconds,
                    outfile_base, *CG_norm_profiles, partition_kind='CG')
        CG_fig.savefig(outfile_base+'_CG.pdf')
        IC_fig.savefig(outfile_base+'_IC.pdf')
    else:
        # no CG data and so the CG profiles should be empty. No need to save them
        length_profiler.write_profile_data(flashes_in_poly.base_date, flashes_in_poly.t_edges_seconds,
                    outfile_base, *IC_norm_profiles, partition_kind='total')
        IC_fig.savefig(outfile_base+'_total.pdf')

# =====
# Energy spectrum plots
# ====
if do_energy_spectra:
    footprint_bin_edges = get_energy_spectrum_bins()
    spectrum_save_file_base = os.path.join(outdir, 'energy_spectrum_{0}_{1}.pdf')
    for flashes, t0, t1 in zip(flashes_series, flashes_in_poly.t_edges[:-1], flashes_in_poly.t_edges[1:]):
        histo, edges = np.histogram(flashes['area'], bins=footprint_bin_edges)
        spectrum_save_file = spectrum_save_file_base.format(t0.strftime('%y%m%d%H%M%S'),
                                                            t1.strftime('%y%m%d%H%M%S'))
        plot_energy_from_area_histogram(histo, edges,
                        save=spectrum_save_file, duration=(t1-t0).total_seconds())

    # =========================================#
    # Energy Spectrum from charge density:     # 
    # =========================================#
    import matplotlib.pyplot as plt
    import matplotlib
    ##Sets color bar for energy spectra plot:
    time_array = []
    for t in flashes_in_poly.t_edges:
        time_array.append(str(t)[11:].replace(':',''))
    time_array = np.asarray(time_array).astype(float)

    norm = matplotlib.colors.Normalize(
         vmin=time_array.min(),
         vmax=time_array.max())

    #ENERGY SPECTRA FOR TOTAL ENERGY:
    cmap_en = plt.cm.gist_heat
    s_m = plt.cm.ScalarMappable(cmap=cmap_en,norm=norm)
    s_m.set_array([])

    spectrum_save_file_base_en = os.path.join(outdir, 'energy_spectrum_estimate_{0}_{1}_new.pdf')
    which_energy = 'total_energy'
    title        = 'Total Energy'
    plot_energies(footprint_bin_edges,
                  time_array, s_m, flashes_series,
                  flashes_in_poly.t_edges,
                  spectrum_save_file_base_en,
                  which_energy,
                  title)

    # ENERGY SPECTRA FOR SPECIFIC ENERGY:
    if do_3d:
        # Flash volume is required to do specific energy
        cmap_specific_en = plt.cm.cubehelix
        s_m2 = plt.cm.ScalarMappable(cmap=cmap_specific_en,norm=norm)
        s_m2.set_array([])

        spectrum_save_file_base_specific = os.path.join(outdir, 'specific_energy_spectrum_estimate_{0}_{1}_new.pdf')
        which_energy = 'specific_energy'
        title        = 'Specific Energy'
        plot_energies(footprint_bin_edges,
                      time_array, s_m2, flashes_series,
                      flashes_in_poly.t_edges,
                      spectrum_save_file_base_specific,
                      which_energy,
                      title)

# =====
# NetCDF grid processing
# =====
field_file = ('flash_extent',
              'flash_init',
              'source',
              'footprint',
              'flashsize_std',
              'total_energy',
              'group_extent',
              'group_init',
              'group_area')
field_names = ('flash_extent_density',
               'flash_centroid_density',
               'event_density',
               'average_flash_area',
               'standard_deviation_flash_area',
               'total_energy',
               'group_extent_density',
               'group_centroid_density',
               'average_group_area',
               )
field_labels = ('Flash extent density (count per pixel)',
                'Flash centroid density (count per pixel)',
                'Event density (count per pixel)',
                'Average flash area (km^2)',
                'Standard deviation of flash area (km^2)',
                'Total radiant energy (J)',
                'Group extent density (count per pixel)',
                'Group centroid density (count per pixel)',
                'Average group area (km^2)',
                )
grid_ranges = ((1, 1000), (1,100), (1, 10000), (50, 100000), (50,100000), 
               (1e-15, 1e-10), (1, 10000), (1,1000), (50, 100000))
field_ids_to_run = (0, 1, 2, 3, 5, 6, 7, 8)
    

def plot_lasso_grid_subset(fig,datalog, t,xedge,yedge,data,grid_lassos,field_name,basedate,grid_range, axis_range, slicer=None):
    from scipy.stats import scoreatpercentile
    from matplotlib.cm import get_cmap
    import matplotlib.colors
    cmap = get_cmap('cubehelix')
    norm=matplotlib.colors.LogNorm()
    
    ax=fig.add_subplot(111)
    ax.set_title(str(t))

    x = (xedge[1:]+xedge[:-1])/2.0
    y = (yedge[1:]+yedge[:-1])/2.0

    xmesh, ymesh = np.meshgrid(x, y)

    assert (xmesh.shape == ymesh.shape == data.shape)

    N = data.size

    # in these grids, Y moves along the zeroth index, X moves along the first index

    orig_shape = data.shape
    a = np.empty((N,), dtype=[('t','f4'), ('lon', 'f4'), ('lat','f4'), (field_name, data.dtype)])
    a['t'] = (t-basedate).total_seconds() + 1.0 # add a bit of padding to make sure it's within the time window
    a['lat'] = ymesh.flatten()
    a['lon'] = xmesh.flatten()
    a[field_name] = data.flatten()

    if slicer is None:
        # Mask out grid cells whose centers are within the polygon
        nonzero = a[field_name] > 0
        good = grid_lassos.filter_mask(a)
        filtered = a[good]
        nonzero_filtered = a[good & nonzero]
        # set up an array for use in pcolor
        masked_nonzero_filtered = a.view(np.ma.MaskedArray)
        masked_nonzero_filtered.shape = data.shape
        masked_nonzero_filtered.mask = (good & nonzero)==False
        
        art = ax.pcolormesh(xedge, yedge,
                            masked_nonzero_filtered[field_name],
                            vmin=grid_range[0], vmax=grid_range[1],
                            cmap=cmap, norm=norm)
    else:
        # EXPERIMENTAL, UNTESTED CODE for slicing grid cells within
        # the polygons instead of masking out by grid cell centers
        for (t0, t1), lassofilter in grid_lassos.poly_lookup.items():
            poly = lassofilter.verts
            in_time = (t >= t0) & (t < t1)
            time_filtered = a[in_time] 
            # Find the polygon valid at the time of the data.
            # There is only one time of data here, and so it will
            # have nonzero size only if it falls within the valid time
            # range of a polygon, and will be one full frame for one
            # polygon valid at that time.
            if time_filtered.size > 0:
                time_filtered.shape = orig_shape
                poly = lassofilter.verts
                sliced_polys, poly_areas = slicer.slice([poly,])
                subquads, frac_areas, q_idxs = sliced_polys[0]
                total_area = poly_areas[0]
                # Fracional areas of each quad. Will use to weight data.
                quad_frac = slicer.quad_frac_from_poly_frac_area(frac_areas,
                    total_area, q_idxs[0], q_idxs[1])
                
                # Use the x,y quad indices to subset the data.
                filtered = time_filtered[q_idxs[0], q_idxs[1]]
                
                # weight the data values by the fractional areas of each sub-polygon.
                filtered[field_data] *= np.asarray(quad_frac, dtype=float)
                
                # patch_vals = np.asarray([time_filtered[field_name][*q_idx] for q_idx in zip(*q_idxs)])
                # patch_vals *= np.asarray(areas, dtype=float)
                
                nonzero = (filtered > 0)
                nonzero_filtered = filtered[nonzero]

                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection
                patches = [Polygon(p, True) for p in subquads]
                art = PatchCollection(patches, norm=norm, cmap=cmap,
                                      vmin=grid_range[0], vmax=grid_range[1])
                ax.add_collection(art)
                
    # ax.plot(lon_range,lat_range,'ok--',alpha=0.5)
    art.set_rasterized(True)
    cbar = fig.colorbar(art)
    cbar.solids.set_rasterized(True) 
    ax.axis(axis_range)

    if nonzero_filtered.size > 0:
        v = filtered[field_name]
        vnz = nonzero_filtered[field_name]
        percentiles = scoreatpercentile(vnz, (5,50,95))
        row = map(str, (t.isoformat(), v.max(), v.sum())+tuple(percentiles))
    else:
        row = map(str, (t, 0 , 0, 0.0, 0.0, 0.0))
    datalog.write(', '.join(row)+'\n')


for field_id in field_ids_to_run:
    nc_filenames = nc_files_from_standard_path(path_to_sort_results,
                        field_file[field_id],
                        min(flashes_in_poly.t_edges),
                        max(flashes_in_poly.t_edges))

    field_name = field_names[field_id]
    grid_range = grid_ranges[field_id]

    fig_outdir = os.path.join(outdir,'grids_{0}'.format(field_name))
    try:
      os.makedirs(fig_outdir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(fig_outdir):
            pass

    gfc = LMAgridFileCollection(nc_filenames, field_name, x_name='longitude', y_name='latitude')
    grid_lassos = flashes_in_poly.grid_lassos
    basedate = flashes_in_poly.base_date

    datalog_name = os.path.join(fig_outdir, '{0}_{1}.csv'.format(field_name, basedate.strftime('%Y%m%d')))
    datalog = open(datalog_name,'w')
    datalog.write('time (ISO), max count per grid box, sum of all grid boxes, 5th percentile, 50th percentile, 95th percentile\n')

    fig=plt.figure(figsize=(12,10))
    lon_range, lat_range = polys_to_bounding_box(flashes_in_poly.polys)
    axis_range = lon_range+lat_range

    # UNCOMMENT TO USE EXPERIMENTAL BLOCK ABOVE
    # mesh = QuadMeshSubset(xedge, yedge, n_neighbors=20)
    # slicer = QuadMeshPolySlicer(mesh)

    for t, xedge, yedge, data in gfc:
        if (t >= t_start) & (t <= t_end):
            plot_lasso_grid_subset(fig,datalog,t,xedge,yedge,data,grid_lassos,field_name,basedate,grid_range, axis_range)
            fig.savefig(os.path.join(fig_outdir, '{0}_{1}.png'.format(field_name, t.strftime('%Y%m%d_%H%M%S'))))
            fig.clf()

    datalog.close()
