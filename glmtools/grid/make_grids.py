""" Gridding of GLM data built on lmatools

"""
import itertools
from functools import partial
import numpy as np
from glmtools.io.mimic_lma import read_flashes
from glmtools.io.glm import GLMDataset
from glmtools.grid.clipping import QuadMeshSubset
from lmatools.grid.make_grids import FlashGridder
from lmatools.grid.fixed import get_GOESR_coordsys
from lmatools.grid.density_to_files import (accumulate_points_on_grid,
    accumulate_points_on_grid_sdev, accumulate_energy_on_grid,
    point_density, extent_density, project,
    flashes_to_frames, flash_count_log, extract_events_for_flashes)
from lmatools.stream.subset import broadcast
import sys
    
class GLMGridder(FlashGridder):
    
    def gridspec_locals(self):
        event_grid_area_fraction_key=self.event_grid_area_fraction_key
        energy_grids=self.energy_grids
        n_frames = self.n_frames
        xedge, yedge, zedge = self.xedge, self.yedge, self.zedge
        dx, dy, dz = self.dx, self.dy, self.dz
        x0 = xedge[0]
        y0 = yedge[0]
        z0 = zedge[0]
        mapProj = self.mapProj
        geoProj = self.geoProj
        return (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj)

    def flash_pipeline_setup(self):
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)
        
        init_density_grid   = np.zeros(grid_shape, dtype='float32')
        extent_density_grid = np.zeros(grid_shape, dtype='float32')
        footprint_grid      = np.zeros(grid_shape, dtype='float32')
        # flashsize_std_grid  = np.zeros(grid_shape, dtype='float32')
        
        all_frames = []
        for i in range(n_frames):
            accum_init_density   = accumulate_points_on_grid(
                init_density_grid[:,:,i], xedge, yedge, label='init')
            accum_extent_density = accumulate_points_on_grid(
                extent_density_grid[:,:,i], xedge, yedge, 
                label='extent', grid_frac_weights=True)
            accum_footprint      = accumulate_points_on_grid(
                footprint_grid[:,:,i], xedge, yedge, 
                label='footprint', grid_frac_weights=True)
            # accum_flashstd       = accumulate_points_on_grid_sdev(
            #     flashsize_std_grid[:,:,i], footprint_grid[:,:,i], xedge, yedge,
            #     label='flashsize_std',  grid_frac_weights=True)

            init_density_target   = point_density(accum_init_density)
            extent_density_target = extent_density(x0, y0, dx, dy, 
                accum_extent_density,
                event_grid_area_fraction_key=event_grid_area_fraction_key)
            mean_footprint_target = extent_density(x0, y0, dx, dy, 
                accum_footprint, weight_key='area',
                event_grid_area_fraction_key=event_grid_area_fraction_key)
            # std_flashsize_target  = extent_density(x0, y0, dx, dy,
            #     accum_flashstd, weight_key='area',
            #     event_grid_area_fraction_key=event_grid_area_fraction_key)

            broadcast_targets = ( 
                project('init_lon', 'init_lat', 'init_alt', mapProj, geoProj, 
                    init_density_target, use_flashes=True),
                project('lon', 'lat', 'alt', mapProj, geoProj, 
                    extent_density_target, use_flashes=False),
                project('lon', 'lat', 'alt', mapProj, geoProj, 
                    mean_footprint_target, use_flashes=False),
                # project('lon', 'lat', 'alt', mapProj, geoProj,
                #     std_flashsize_target, use_flashes=False),
                )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(extract_events_for_flashes(spew_to_density_types))

        frame_count_log = flash_count_log(self.flash_count_logfile)
        
        framer = flashes_to_frames(self.t_edges_seconds, all_frames, 
                     time_key='start', time_edges_datetime=self.t_edges, 
                     flash_counter=frame_count_log)

        outgrids = (init_density_grid, extent_density_grid,
            footprint_grid, 
            # flashsize_std_grid,
            )
        return outgrids, framer

    def event_pipeline_setup(self):
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)
        event_density_grid  = np.zeros(grid_shape, dtype='float32')
        total_energy_grid   = np.zeros(grid_shape, dtype='float32')

        all_frames = []
        for i in range(n_frames):
            accum_event_density  = accumulate_points_on_grid(
                event_density_grid[:,:,i], xedge, yedge,
                label='event', grid_frac_weights=True)
            event_density_target  = extent_density(x0, y0, dx, dy,
                accum_event_density,
                event_grid_area_fraction_key=event_grid_area_fraction_key)
                
            # total_energy is built from split_event_energy, which has
            # already divided up the event energy into the sub-event
            # corresponding to each pixel. We don't need to weight by the
            # grid fractional area. We just need to sum the  'power' variable
            # which mimic_lma assigns the values of split_event_energy.
            accum_total_energy   = accumulate_energy_on_grid(
                total_energy_grid[:,:,i], xedge, yedge,
                label='total_energy',  grid_frac_weights=False)
            total_energy_target = point_density(accum_total_energy,
                weight_key='power', weight_flashes=False)

            broadcast_targets = (
                 project('lon', 'lat', 'alt', mapProj, geoProj,
                     event_density_target, use_flashes=False),
                 project('lon', 'lat', 'alt', mapProj, geoProj,
                     total_energy_target, use_flashes=False),
            )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(extract_events_for_flashes(spew_to_density_types))

        frame_count_log = flash_count_log(self.flash_count_logfile)

        framer = flashes_to_frames(self.t_edges_seconds, all_frames,
                     time_key='start', time_edges_datetime=self.t_edges,
                     flash_counter=frame_count_log)

        outgrids = (event_density_grid, total_energy_grid)
        return outgrids, framer

    def pipeline_setup(self):
        """
        
        """
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)

        flash_outgrids, flash_framer = self.flash_pipeline_setup()
        (init_density_grid, extent_density_grid, footprint_grid,
            # flashsize_std_grid
            ) = flash_outgrids

        # From a data structure point of view, there is no difference
        # between the group and flash grids. Later, if there are differnces,
        # just copy the flash_pipeline_setup method and start modifying.
        group_outgrids, group_framer = self.flash_pipeline_setup()
        (group_centroid_density_grid, group_extent_density_grid, 
            group_footprint_grid, 
            # groupsize_std_grid
            ) = group_outgrids

        event_outgrids, event_framer = self.event_pipeline_setup()
        event_density_grid, total_energy_grid = event_outgrids

        self.outgrids = (
            extent_density_grid,
            init_density_grid,
            event_density_grid,
            footprint_grid,
            # flashsize_std_grid,
            total_energy_grid,
            group_extent_density_grid, 
            group_centroid_density_grid, 
            group_footprint_grid, 
            # groupsize_std_grid
            )
        self.outgrids_3d = None

        all_framers = {'flash': flash_framer,
                       'group': group_framer,
                       'event': event_framer,
                      }

        self.framer = all_framers
        
    def output_setup(self):
        """
        For each of the grids of interest in self.outgrids, set up the
        outfile names, units, etc. These are all the metadata that go with the
        actual values on the grids.
        """
        
        energy_grids = self.energy_grids
        spatial_scale_factor = self.spatial_scale_factor        
        dx, dy, dz = self.dx, self.dy, self.dz
            
        if self.proj_name=='latlong':
            density_units = "grid"
        elif self.proj_name == 'geos':
            density_units = '{0:7d} radians^2'.format(int(dx*dy))
        else:
            density_units = "{0:5.1f} km^2".format(dx*spatial_scale_factor * dy*spatial_scale_factor).lstrip()
        time_units = "{0:5.1f} min".format(self.frame_interval/60.0).lstrip()
        density_label = 'Count per ' + density_units + " pixel per "+ time_units
    
        self.outfile_postfixes = ('flash_extent.nc',
                                  'flash_init.nc',
                                  'source.nc',
                                  'footprint.nc',
                                  # 'flashsize_std.nc',
                                  'total_energy.nc',
                                  'group_extent.nc',
                                  'group_init.nc',
                                  'group_area.nc',)
        self.outfile_postfixes_3d = None
                                                            
        self.field_names = ('flash_extent_density',
                       'flash_centroid_density',
                       'event_density',
                       'average_flash_area',
                       # 'standard_deviation_flash_area',
                       'total_energy',
                       'group_extent_density',
                       'group_centroid_density',
                       'average_group_area',
                       )
    
        self.field_descriptions = ('Flash extent density',
                            'Flash initiation density',
                            'Event density',
                            'Average flash area',
                            # 'Standard deviation of flash area',
                            'Total radiant energy',
                            'Group extent density',
                            'Group centroid density',
                            'Average group area',
                            )
        
        self.field_units = (
            density_label,
            density_label,
            density_label,
            "km^2 per flash",
            # "km^2",
            "J per flash",
            density_label,
            density_label,
            "km^2 per group",
            )
        self.field_units_3d = None
        
        self.outformats = ('f',) * 8
        self.outformats_3d = ('f',) * 8
        
    
    def process_flashes(self, glm, lat_bnd=None, lon_bnd=None, 
                        min_points_per_flash=None, min_groups_per_flash=None,
                        clip_events=False, fixed_grid=False,
                        nadir_lon=None, corner_pickle=None):
        self.min_points_per_flash = min_points_per_flash
        if min_points_per_flash is None:
            # the FlashGridder class from lmatools needs an int to be able to
            # write files but otherwise the GLM classes need None to avoid
            # processing the minimum points per flash criteria.
            self.min_points_per_flash = 1
        self.min_groups_per_flash = min_groups_per_flash
        if min_groups_per_flash is None:
            self.min_groups_per_flash = 1
        # interpret x_bnd and y_bnd as lon, lat
        read_flashes(glm, self.framer, base_date=self.t_ref, 
                     min_events=min_points_per_flash,
                     min_groups=min_groups_per_flash,
                     lon_range=lon_bnd, lat_range=lat_bnd,
                     clip_events=clip_events, fixed_grid=fixed_grid,
                     nadir_lon=nadir_lon)

def subdivide_bnd(bnd, delta, s=8):
    """
    Subdivide a range (edges of the span) into s equal length
    segments. delta is the spacing of grid boxes within bnd.
    The last segment may be slightly longer to accomodate a number
    of grid boxes that doesn't divide evenly into the number of segments

    Example
    -------
    s = 3
    bnd, delta = (0,4), 0.5
    |  |  |  |  |  |  |  |  |
    0     1     2     3     4
    n = 8

    """
    w = bnd[1] - bnd[0]
    # Number of elements
    n = int(w/delta)
    # Numbr of elements in each segment
    print(n, s)
    dn = int(n/s)

    s_edges = np.arange(s+1, dtype='f8')*delta*dn + bnd[0]
    s_edges[-1] = bnd[1]
    return s_edges

def subdivided_fixed_grid(kwargs, process_flash_kwargs, out_kwargs, s=1,
    x_pad = 100*28.0e-6, y_pad = 100*28.0e-6):
    """

    Generator function to turn a single set of keyword arguments to
    grid_GLM_flashes into an s by s block of keyword arguments.

    x_pad and y_pad are padding in fixed grid coordinates used to increase
    lon_bnd and lat_bnd. Default is the equivalent of 100 km at nadir.
    When flashes are subset for processing on a target grid, lon_bnd and
    lat_bnd are used to filter flash centroids. A flash with a centroid near
    the edge of the target grid may have events within the grid, so we want
    to capture that flash too.

    Yields (i,j) kwargs_ij, process_flash_kwargs_ij for each i,j subgrid.
    """

    x_sub_bnd = subdivide_bnd(kwargs['x_bnd'], kwargs['dx'], s=s)
    y_sub_bnd = subdivide_bnd(kwargs['y_bnd'], kwargs['dy'], s=s)
    print(x_sub_bnd, y_sub_bnd)
    nadir_lon = process_flash_kwargs['nadir_lon']
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)


    for i, j in itertools.product(range(s), range(s)):
        kwargsij = kwargs.copy()
        prockwargsij = process_flash_kwargs.copy()
        x_bnd_i = x_sub_bnd[i:i+2].copy()
        y_bnd_j = y_sub_bnd[j:j+2].copy()
        # Make a copy of this and use it as target grid specs before
        # we modify to use as the lon/lat bounding box
        kwargsij['x_bnd'], kwargsij['y_bnd'] = x_bnd_i.copy(), y_bnd_j.copy()
        x_bnd_i[0] -= x_pad
        x_bnd_i[1] += x_pad
        y_bnd_j[0] -= y_pad
        y_bnd_j[1] += y_pad
        lon_bnd, lat_bnd, alt_bnd = grs80lla.fromECEF( *geofixcs.toECEF(
                                        x_bnd_i, y_bnd_j, x_bnd_i*0.0))
        kwargsij['x_bnd'], kwargsij['y_bnd'] = x_bnd_i, y_bnd_j
        # Both bounds are inf if the fixed grid location is off the edge of
        # the earth. Later, when flashes are subset by glm.subset_flashes,
        # this causes the subset to be empty.
        # Work around that by setting the left bound to -infinity
        if lon_bnd[0] == np.inf: lon_bnd[0] *= -1
        if lat_bnd[0] == np.inf: lat_bnd[0] *= -1
        prockwargsij['lon_bnd'], prockwargsij['lat_bnd'] = None, None
        # prockwargsij['lon_bnd'], prockwargsij['lat_bnd'] = lon_bnd, lat_bnd
        # This is an instance of the coordinate systems class, and it doesn't
        # pickle. Recreate on the subprocess later on.
        # kwargsij['pixel_coords'] = None
        
        outfile_prefix_base = out_kwargs['output_filename_prefix']
        outfile_prefix_base_ij = outfile_prefix_base + '-'
        outfile_prefix_base_ij += '{0:02d}-{1:02d}'.format(i,j)
        out_kwargs_ij = out_kwargs.copy()
        out_kwargs_ij['output_filename_prefix'] = outfile_prefix_base_ij
        # redefine this function later on the subprocess
        # out_kwargs_ij['output_writer'] = None
        
        print(i,j, x_bnd_i, y_bnd_j, lon_bnd, lat_bnd)

        yield (i, j), kwargsij, prockwargsij, out_kwargs_ij

# @profile
def grid_GLM_flashes(GLM_filenames, start_time, end_time, **kwargs):
    """ Grid GLM data that has been converted to an LMA-like array format.
        
        Assumes that GLM data are being gridded on a lat, lon grid.
        
        Keyword arguments to this function
        are those to the FlashGridder class and its functions.
    """
    
    kwargs['do_3d'] = False
    
    # Used only for the fixed grid at the moment
    subdivide_grid = kwargs.pop('subdivide', 4)

    process_flash_kwargs = {}
    for prock in ('min_points_per_flash','min_groups_per_flash',
                  'clip_events', 'fixed_grid', 'nadir_lon', 'corner_pickle'):
        # interpret x_bnd and y_bnd as lon, lat
        if prock in kwargs:
            process_flash_kwargs[prock] = kwargs.pop(prock)
    # need to also pass these kwargs through to the gridder for grid config.
    if 'clip_events' in process_flash_kwargs:
        kwargs['event_grid_area_fraction_key'] = 'mesh_frac'

    out_kwargs = {}
    for outk in ('outpath', 'output_writer', 'output_writer_3d',
                 'output_kwargs', 'output_filename_prefix'):
        if outk in kwargs:
            out_kwargs[outk] = kwargs.pop(outk)

    if kwargs['proj_name'] == 'latlong':
        process_flash_kwargs['lon_bnd'] = kwargs['x_bnd']
        process_flash_kwargs['lat_bnd'] = kwargs['y_bnd']
        subgrids = [((0, 0), kwargs, process_flash_kwargs, out_kwargs)]
    elif 'fixed_grid' in process_flash_kwargs:
        subgrids = subdivided_fixed_grid(kwargs, process_flash_kwargs, 
                                         out_kwargs, s=subdivide_grid)
    else:
        # working with ccd pixels or a projection, so no known lat lon bnds
        process_flash_kwargs['lon_bnd'] = None
        process_flash_kwargs['lat_bnd'] = None
        subgrids = [((0, 0), kwargs, process_flash_kwargs, out_kwargs)]

    this_proc_each_grid = partial(proc_each_grid, start_time=start_time,
        end_time=end_time, GLM_filenames=GLM_filenames)
    outputs = pool.map(this_proc_each_grid, subgrids)
    return outputs
    
from concurrent.futures import ProcessPoolExecutor
pool = ProcessPoolExecutor(max_workers=4)

# @profile
def proc_each_grid(subgrid, start_time=None, end_time=None, 
    GLM_filenames=None):
    subgridij, kwargsij, process_flash_kwargs_ij, out_kwargs_ij = subgrid
    
    print("out kwargs are", out_kwargs_ij)
        
    # These should all be independent at this point and can parallelize
    print ('gridder kwargs for subgrid {0} are'.format(subgridij), kwargsij)
    gridder = GLMGridder(start_time, end_time, **kwargsij)

    if 'clip_events' in process_flash_kwargs_ij:
        xedge,yedge=np.meshgrid(gridder.xedge,gridder.yedge)
        mesh = QuadMeshSubset(xedge, yedge, n_neighbors=16*10, regular=True)
        # import pickle
        # with open('/data/LCFA-production/L1b/mesh_subset.pickle', 'wb') as f:
            # pickle.dump(mesh, f)
        process_flash_kwargs_ij['clip_events'] = mesh
    for filename in GLM_filenames:
        # Could create a cache of GLM objects by filename here.
        print("Processing {0}".format(filename))
        print('process flash kwargs are', process_flash_kwargs_ij)
        sys.stdout.flush()
        glm = GLMDataset(filename)
        # Pre-load the whole dataset, as recommended by the xarray docs.
        # This saves an absurd amount of time (factor of 80ish) in
        # grid.split_events.replicate_and_split_events
        glm.dataset.load()
        gridder.process_flashes(glm, **process_flash_kwargs_ij)
        glm.dataset.close()
        del glm

    # output = gridder.write_grids(**out_kwargs_ij)
    return subgridij
