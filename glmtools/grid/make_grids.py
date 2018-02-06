""" Gridding of GLM data built on lmatools

"""
import numpy as np
from glmtools.io.mimic_lma import read_flashes
from glmtools.io.glm import GLMDataset
from glmtools.grid.clipping import QuadMeshSubset
from lmatools.grid.make_grids import FlashGridder
from lmatools.grid.fixed import get_GOESR_coordsys
from lmatools.grid.density_to_files import (accumulate_points_on_grid,
    accumulate_points_on_grid_sdev, accumulate_energy_on_grid,
    accumulate_points_on_grid_3d, accumulate_points_on_grid_sdev_3d,
    accumulate_energy_on_grid_3d,
    point_density, extent_density, point_density_3d, extent_density_3d, project,
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
        flashsize_std_grid  = np.zeros(grid_shape, dtype='float32')
        
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
            accum_flashstd       = accumulate_points_on_grid_sdev(
                flashsize_std_grid[:,:,i], footprint_grid[:,:,i], xedge, yedge, 
                label='flashsize_std',  grid_frac_weights=True)

            init_density_target   = point_density(accum_init_density)
            extent_density_target = extent_density(x0, y0, dx, dy, 
                accum_extent_density,
                event_grid_area_fraction_key=event_grid_area_fraction_key)
            mean_footprint_target = extent_density(x0, y0, dx, dy, 
                accum_footprint, weight_key='area',
                event_grid_area_fraction_key=event_grid_area_fraction_key)
            std_flashsize_target  = extent_density(x0, y0, dx, dy, 
                accum_flashstd, weight_key='area',
                event_grid_area_fraction_key=event_grid_area_fraction_key)

            broadcast_targets = ( 
                project('init_lon', 'init_lat', 'init_alt', mapProj, geoProj, 
                    init_density_target, use_flashes=True),
                project('lon', 'lat', 'alt', mapProj, geoProj, 
                    extent_density_target, use_flashes=False),
                project('lon', 'lat', 'alt', mapProj, geoProj, 
                    mean_footprint_target, use_flashes=False),
                project('lon', 'lat', 'alt', mapProj, geoProj, 
                    std_flashsize_target, use_flashes=False),
                )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(extract_events_for_flashes(spew_to_density_types))

        frame_count_log = flash_count_log(self.flash_count_logfile)
        
        framer = flashes_to_frames(self.t_edges_seconds, all_frames, 
                     time_key='start', time_edges_datetime=self.t_edges, 
                     flash_counter=frame_count_log)

        outgrids = (init_density_grid, extent_density_grid,
            footprint_grid, flashsize_std_grid)
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
            accum_total_energy   = accumulate_energy_on_grid(
                total_energy_grid[:,:,i], xedge, yedge,
                label='total_energy',  grid_frac_weights=True)

            event_density_target  = extent_density(x0, y0, dx, dy,
                accum_event_density,
                event_grid_area_fraction_key=event_grid_area_fraction_key)
            total_energy_target = extent_density(x0, y0, dx, dy,
                accum_total_energy, weight_key='total_energy',
                event_grid_area_fraction_key=event_grid_area_fraction_key)

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
            flashsize_std_grid) = flash_outgrids

        event_outgrids, event_framer = self.event_pipeline_setup()
        event_density_grid, total_energy_grid = event_outgrids

        self.outgrids = (
            extent_density_grid,
            init_density_grid,
            event_density_grid,
            footprint_grid,
            flashsize_std_grid,
            total_energy_grid,
            )
        self.outgrids_3d = None

        all_framers = {'flash': flash_framer,
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
                                  'flashsize_std.nc',
                                  'total_energy.nc')
        self.outfile_postfixes_3d = None
                                                            
        self.field_names = ('flash_extent_density',
                       'flash_centroid_density',
                       'event_density',
                       'average_flash_area',
                       'standard_deviation_flash_area',
                       'total_energy')
    
        self.field_descriptions = ('Flash extent density',
                            'Flash initiation density',
                            'Event density',
                            'Average flash area',
                            'Standard deviation of flash area',
                            'Total radiant energy')
        
        self.field_units = (
            density_label,
            density_label,
            density_label,
            "km^2 per flash",
            "km^2",
            "J per flash",
            )
        self.field_units_3d = None
        
        self.outformats = ('f', 'f', 'f', 'f', 'f', 'f')
        self.outformats_3d = ('f', 'f', 'f', 'f', 'f', 'f')
        
    
    def process_flashes(self, glm, lat_bnd=None, lon_bnd=None, 
                        min_points_per_flash=1, min_groups_per_flash=1,
                        clip_events=False, fixed_grid=False,
                        nadir_lon=None):
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
                     min_events=self.min_points_per_flash,
                     min_groups=self.min_groups_per_flash,
                     lon_range=lon_bnd, lat_range=lat_bnd,
                     clip_events=clip_events, fixed_grid=fixed_grid,
                     nadir_lon=nadir_lon)

        
def grid_GLM_flashes(GLM_filenames, start_time, end_time, **kwargs):
    """ Grid GLM data that has been converted to an LMA-like array format.
        
        Assumes that GLM data are being gridded on a lat, lon grid.
        
        Keyword arguments to this function
        are those to the FlashGridder class and its functions.
    """
    
    kwargs['do_3d'] = False
    
    process_flash_kwargs = {}
    for prock in ('min_points_per_flash','min_groups_per_flash',
                  'clip_events', 'fixed_grid', 'nadir_lon'):
        # interpret x_bnd and y_bnd as lon, lat
        if prock in kwargs:
            process_flash_kwargs[prock] = kwargs.pop(prock)
    # need to also pass these kwargs through to the gridder for grid config.
    if kwargs['proj_name'] == 'latlong':
        process_flash_kwargs['lon_bnd'] = kwargs['x_bnd']
        process_flash_kwargs['lat_bnd'] = kwargs['y_bnd']
    elif 'fixed_grid' in process_flash_kwargs:
        nadir_lon = process_flash_kwargs['nadir_lon']
        geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
        lon_bnd, lat_bnd, alt_bnd = grs80lla.fromECEF( *geofixcs.toECEF(
            kwargs['x_bnd'], kwargs['y_bnd'], kwargs['x_bnd']*0.0))
        process_flash_kwargs['lon_bnd'] = lon_bnd
        process_flash_kwargs['lat_bnd'] = lat_bnd
    else:
        # working with ccd pixels or a projection, so no known lat lon bnds
        process_flash_kwargs['lon_bnd'] = None
        process_flash_kwargs['lat_bnd'] = None
    
    if 'clip_events' in process_flash_kwargs:
        kwargs['event_grid_area_fraction_key'] = 'mesh_frac'
            

    out_kwargs = {}
    for outk in ('outpath', 'output_writer', 'output_writer_3d',
                 'output_kwargs', 'output_filename_prefix'):
        if outk in kwargs:
            out_kwargs[outk] = kwargs.pop(outk)
    
    gridder = GLMGridder(start_time, end_time, **kwargs)

    if 'clip_events' in process_flash_kwargs:
        xedge,yedge=np.meshgrid(gridder.xedge,gridder.yedge)
        mesh = QuadMeshSubset(xedge, yedge, n_neighbors=16*10)
        process_flash_kwargs['clip_events'] = mesh
    for filename in GLM_filenames:
        print("Processing {0}".format(filename))
        sys.stdout.flush()
        glm = GLMDataset(filename)
        gridder.process_flashes(glm, **process_flash_kwargs)

    output = gridder.write_grids(**out_kwargs)
    return output    

