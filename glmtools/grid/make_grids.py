""" Gridding of GLM data built on lmatools

"""

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import itertools
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from glmtools.io.mimic_lma import read_flashes
from glmtools.io.glm import GLMDataset
from glmtools.grid.clipping import QuadMeshSubset
from lmatools.grid.make_grids import FlashGridder
from lmatools.grid.fixed import get_GOESR_coordsys
from lmatools.grid.density_to_files import (accumulate_points_on_grid,
    accumulate_points_on_grid_sdev, accumulate_energy_on_grid,
    point_density, extent_density, project, accumulate_minimum_on_grid,
    flashes_to_frames, flash_count_log, extract_events_for_flashes)
from .accumulate import (select_dataset, accumulate_var_on_grid_direct_idx,
    accumulate_minvar_on_grid_direct_idx)
from lmatools.stream.subset import broadcast
import sys

class GLMGridder(FlashGridder):
    """ Subclass of lmatools.FlashGridder specialized for gridding GLM data. Sets up
    an accumulation pipeline that receives flash, group, or event data and flows those
    data to the appropriate target grids.

    Methods:
    gridspec_locals -- Return a tuple of instance attributes that describe the grid
        configuration.
    pipeline_setup -- Calls and stores the result of the two methods below
    flash_pipeline_setup -- Creates the flash- and group-level GLM grids and sets up the
        accumulation pipeline for each grid type
    event_pipeline_setup -- Creates the event-level GLM grids and sets up the
        accumulation pipeline for each grid type
    output_setup -- Sets up the NetCDF filenames, variable names, etc. for use by the
        output writer.
    process_flashes -- Read data from a glmtools.io.glm.GLMDataset instance and push it
        into the accumulation pipeline

    """
    def gridspec_locals(self):
        """  Return a tuple of instance attributes that describe the grid configuration.

        Arguments: none.

        Returns, in order as a tuple:
        event_grid_area_fraction_key -- column name in event data table that gives
            the fraction of the target grid cell covered by the event.
        energy_grids -- names of the grids that are some measure of energy
        n_frames -- number of time intervals in this grid
        xedge, yedge, zedge -- edges (not centers) of the grid cells
        dx, dy, dz -- spacing between the grid cells
        x0, y0, z0 -- first value of xedge, yedge, zedge
        mapProj -- lmatools.CoordinateSystem instance for the target grid.
        geoProj -- lmatools.CoordinateSystem.GeographicSystem instance for use in
            converting longitude/latitude/altitude coordinates
        """
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

    def group_pipeline_setup(self):
        # From a data structure point of view, there is no difference
        # between the group and flash grids. Later, if there are differnces,
        # just copy the flash_pipeline_setup method and start modifying.
        return self.flash_pipeline_setup()

    def flash_pipeline_setup(self):
        """ Create target grids and set up the flash accumulation pipeline. Also used
        to set up an equivalent group accumulation pipeline.

        Arguments: none

        Returns:
        outgrids -- tuple of grids (numpy arrays) for the target grids
        framer -- first stage of the pipeline that routes flashes to the correct
            time window for accumulation.
        """
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
        """ Create target grids and set up the event accumulation pipeline.

        Arguments: none

        Returns:
        outgrids -- tuple of grids (numpy arrays) for the target grids
        framer -- first stage of the pipeline that routes flashes to the correct
            time window for accumulation.
        """
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
        """ Create target grids and set up the flash, group and event accumulation
        pipelines. Each pipeline starts by subdividing the flash dataset into the correct
        time frames, then broadcasts those time chunks to a separate pipeline for each
        grid subtype. Those pipelines typically then project the data onto the target
        grid, calculate point or extent density, and accumulate on the target grid
        (weighting by fractional coverage of the target grid in some cases).

        Arguments: none.
        Returns: none.

        Sets the following instance attributes:
        outgrids -- tuple of grid types (numpy arrays)
        outgrids_3d -- None
        framer -- dictionary, with keys 'flash', 'group', and 'event' pointing to the
            pipeline inlet for each.
        """
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)

        flash_outgrids, flash_framer = self.flash_pipeline_setup()
        (init_density_grid, extent_density_grid, footprint_grid,
            min_flash_area_grid, # flashsize_std_grid
            ) = flash_outgrids

        group_outgrids, group_framer = self.group_pipeline_setup()
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
            min_flash_area_grid,
            )
        self.outgrids_3d = None

        all_framers = {'flash': flash_framer,
                       'group': group_framer,
                       'event': event_framer,
                      }

        self.framer = all_framers

    def output_setup(self):
        """ Set up grid names and variable types for output.

        Arguments: none.
        Returns: none.

        Sets the following instance attributes, which parallel self.outgrids as set
        by self.setup_pipeline:
        outfile_postfixes -- last part of each NetCDF filename
        field_names -- variable names within the NetCDF file for each grid
        field_descriptions -- plain language description of each grid
        field_units -- units for each grid
        outformats -- NetCDF variable type codes for each grid
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
                                  'group_area.nc',
                                  'flash_area_min.nc')
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
                       'minimum_flash_area',
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
                            'Minimum flash area',
                            )

        # In some use cases, it's easier to calculate totals (for area or
        # energy) and then divide at the end. This dictionary maps numerator
        # to denominator, with an index corresponding to self.outgrids.
        # The avearge is then calculated on output with numerator_out =
        # numerator/denominator. For example to calculate average energy
        # instead of total energy:
        #    self.divide_grids[6]=0
        # and change the labels in field_names, etc. to read as averages
        # instead of totals.
        self.divide_grids = {}

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
            "km^2",
            )
        self.field_units_3d = None

        self.outformats = ('f',) * len(self.field_units)
        self.outformats_3d = ('f',) * len(self.field_units)


    def process_flashes(self, glm, lat_bnd=None, lon_bnd=None,
                        x_bnd=None, y_bnd=None,
                        min_points_per_flash=None, min_groups_per_flash=None,
                        clip_events=False, fixed_grid=False,
                        nadir_lon=None, corner_pickle=None):
        """ Read data from a glmtools.io.glm.GLMDataset instance and push it into the
        accumulation pipeline

        Arguments:
        glm -- a glmtools.io.glm.GLMDataset instance

        Keyword arguments:
        corner_pickle: currently unused.
        clip_events, fixed_grid, nadir_lon: passed directly to
            glmtools.io.mimic_lma.read_flashes.
        lat_bnd, lon_bnd, x_bnd, y_bnd -- passed to lat_range, lon_range, x_range,
            y_range in glmtools.io.mimic_lma.read_flashes.
        min_groups_per_flash and min_points_per_flash are passed to min_groups
            and min_events in glmtools.io.mimic_lma.read_flashes.

        """
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
                     x_range=x_bnd, y_range=y_bnd,
                     clip_events=clip_events, fixed_grid=fixed_grid,
                     nadir_lon=nadir_lon)

class GLMlutGridder(GLMGridder):
    def event_pipeline_setup(self):
        """ Create target grids and set up the event accumulation pipeline for events
        that have been pre-aggregated using an event lookup table.

        Arguments: none

        Returns:
        outgrids -- tuple of grids (numpy arrays) for the target grids
        framer -- first stage of the pipeline that routes flashes to the correct
            time window for accumulation.
        """
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)
        event_density_grid  = np.zeros(grid_shape, dtype='float32')
        total_energy_grid   = np.zeros(grid_shape, dtype='float32')

        all_frames = []
        for i in range(n_frames):
            accum_event_density  = accumulate_energy_on_grid(
                event_density_grid[:,:,i], xedge, yedge,
                label='event extent', grid_frac_weights=False)
            event_density_target  = point_density(accum_event_density,
                weight_key='lutevent_count', weight_flashes=False)

            # total_energy is built from split_event_energy, which has
            # already divided up the event energy into the sub-event
            # corresponding to each pixel. We don't need to weight by the
            # grid fractional area. We just need to sum the  'power' variable
            # which mimic_lma assigns the values of split_event_energy.
            accum_total_energy   = accumulate_energy_on_grid(
                total_energy_grid[:,:,i], xedge, yedge,
                label='total energy',  grid_frac_weights=False)
            total_energy_target = point_density(accum_total_energy,
                weight_key='power', weight_flashes=False)

            broadcast_targets = (
                 project('lon', 'lat', 'alt', mapProj, geoProj,
                     event_density_target, use_flashes=False),
                 project('lon', 'lat', 'alt', mapProj, geoProj,
                     total_energy_target, use_flashes=False),
            )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(spew_to_density_types)

        frame_count_log = flash_count_log(self.flash_count_logfile)

        framer = flashes_to_frames(self.t_edges_seconds, all_frames,
                     time_key='time', time_edges_datetime=self.t_edges,
                     flash_counter=frame_count_log, do_events='time')

        outgrids = (event_density_grid, total_energy_grid)
        return outgrids, framer

    def flash_pipeline_setup(self):
        """ Create target grids and set up the flash accumulation pipeline. Also used
        to set up an equivalent group accumulation pipeline. Actually uses event-level
        data that have been pre-aggregated using an event lookup table.

        Arguments: none

        Returns:
        outgrids -- tuple of grids (numpy arrays) for the target grids
        framer -- first stage of the pipeline that routes flashes to the correct
            time window for accumulation.
        """
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)

        init_density_grid   = np.zeros(grid_shape, dtype='float32')
        extent_density_grid = np.zeros(grid_shape, dtype='float32')
        footprint_grid      = np.zeros(grid_shape, dtype='float32')
        min_area_grid       = np.zeros(grid_shape, dtype='float32')

        all_frames = []
        for i in range(n_frames):

            # accum_total_energy   = accumulate_energy_on_grid(
#                 total_energy_grid[:,:,i], xedge, yedge,
#                 label='total_energy',  grid_frac_weights=False)
#             total_energy_target = point_density(accum_total_energy,
#                 weight_key='power', weight_flashes=False)
#
            accum_init_density   = accumulate_points_on_grid(
                init_density_grid[:,:,i], xedge, yedge, label='init')
            # accum_extent_density = accumulate_energy_on_grid(
            #     extent_density_grid[:,:,i], xedge, yedge,
            #     label='flash extent', grid_frac_weights=False)
            accum_extent_density = accumulate_var_on_grid_direct_idx(
                    extent_density_grid[:,:,i],
                    'lutevent_flash_count', 'mesh_xi', 'mesh_yi')
            # accum_footprint      = accumulate_energy_on_grid(
            #     footprint_grid[:,:,i], xedge, yedge,
            #     label='flash area', grid_frac_weights=False)
            accum_footprint = accumulate_var_on_grid_direct_idx(
                    footprint_grid[:,:,i],
                    'lutevent_total_flash_area', 'mesh_xi', 'mesh_yi')
            # accum_min_area       = accumulate_minimum_on_grid(
            #     min_area_grid[:,:,i], xedge, yedge,
            #     label='min flash area', grid_frac_weights=False)
            accum_min_area = accumulate_minvar_on_grid_direct_idx(
                     min_area_grid[:,:,i],
                    'lutevent_min_flash_area', 'mesh_xi', 'mesh_yi')

            init_density_target   = point_density(accum_init_density)
            # extent_density_target = point_density(accum_extent_density,
            #     weight_key='lutevent_flash_count', weight_flashes=False)
            # mean_footprint_target = point_density(accum_footprint,
                # weight_key='lutevent_total_flash_area', weight_flashes=False)
            # min_area_target = point_density(accum_min_area,
                # weight_key='lutevent_min_flash_area', weight_flashes=False)

            broadcast_targets = (
                project('init_lon', 'init_lat', 'init_alt', mapProj, geoProj,
                    init_density_target, use_flashes=True),
                # project('lon', 'lat', 'alt', mapProj, geoProj,
                #     extent_density_target, use_flashes=False),
                select_dataset(accum_extent_density, use_event_data=True),
                # project('lon', 'lat', 'alt', mapProj, geoProj,
                #     mean_footprint_target, use_flashes=False),
                select_dataset(accum_footprint, use_event_data=True),
                # project('lon', 'lat', 'alt', mapProj, geoProj,
                #     min_area_target, use_flashes=False),
                select_dataset(accum_min_area, use_event_data=True),
                )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(spew_to_density_types)

        frame_count_log = flash_count_log(self.flash_count_logfile)

        framer = flashes_to_frames(self.t_edges_seconds, all_frames,
                     time_key='start', time_edges_datetime=self.t_edges,
                     flash_counter=frame_count_log, do_events='time')

        outgrids = (init_density_grid, extent_density_grid,
            footprint_grid, min_area_grid,
            )
        return outgrids, framer

    def group_pipeline_setup(self):
        """ Create target grids and set up the flash accumulation pipeline. Also used
        to set up an equivalent group accumulation pipeline. Actually uses event-level
        data that have been pre-aggregated using an event lookup table.

        Arguments: none

        Returns:
        outgrids -- tuple of grids (numpy arrays) for the target grids
        framer -- first stage of the pipeline that routes flashes to the correct
            time window for accumulation.
        """
        (event_grid_area_fraction_key, energy_grids, n_frames,
            xedge, yedge, zedge, dx, dy, dz, x0, y0, z0, mapProj, geoProj) = (
            self.gridspec_locals())

        grid_shape = (xedge.shape[0]-1, yedge.shape[0]-1, n_frames)

        init_density_grid   = np.zeros(grid_shape, dtype='float32')
        extent_density_grid = np.zeros(grid_shape, dtype='float32')
        footprint_grid      = np.zeros(grid_shape, dtype='float32')

        all_frames = []
        for i in range(n_frames):

            # accum_total_energy   = accumulate_energy_on_grid(
#                 total_energy_grid[:,:,i], xedge, yedge,
#                 label='total_energy',  grid_frac_weights=False)
#             total_energy_target = point_density(accum_total_energy,
#                 weight_key='power', weight_flashes=False)
#
            accum_init_density   = accumulate_points_on_grid(
                init_density_grid[:,:,i], xedge, yedge, label='init')
            accum_extent_density = accumulate_energy_on_grid(
                extent_density_grid[:,:,i], xedge, yedge,
                label='group extent', grid_frac_weights=False)
            accum_footprint      = accumulate_energy_on_grid(
                footprint_grid[:,:,i], xedge, yedge,
                label='group area', grid_frac_weights=False)

            init_density_target   = point_density(accum_init_density)
            extent_density_target = point_density(accum_extent_density,
                weight_key='lutevent_group_count', weight_flashes=False)
            mean_footprint_target = point_density(accum_footprint,
                weight_key='lutevent_total_group_area', weight_flashes=False)

            broadcast_targets = (
                project('init_lon', 'init_lat', 'init_alt', mapProj, geoProj,
                    init_density_target, use_flashes=True),
                project('lon', 'lat', 'alt', mapProj, geoProj,
                    extent_density_target, use_flashes=False),
                project('lon', 'lat', 'alt', mapProj, geoProj,
                    mean_footprint_target, use_flashes=False),
                )
            spew_to_density_types = broadcast( broadcast_targets )

            all_frames.append(spew_to_density_types)

        frame_count_log = flash_count_log(self.flash_count_logfile)

        framer = flashes_to_frames(self.t_edges_seconds, all_frames,
                     time_key='start', time_edges_datetime=self.t_edges,
                     flash_counter=frame_count_log, do_events='time')

        outgrids = (init_density_grid, extent_density_grid,
            footprint_grid,
            )
        return outgrids, framer

    def output_setup(self, *args, **kwargs):
        super(GLMlutGridder, self).output_setup(*args, **kwargs)
        # In the LUT gridding we calculate totals for area and then divide at
        # the end. This dictionary maps numerator
        # to denominator, with an index corresponding to self.outgrids.
        # The avearge is then calculated on output with numerator_out =
        # numerator/denominator just before the grid is written.
        log.info('Setting up to divide area grids by extent density grids')
        # self.field_names = ('flash_extent_density', 0
        #                'flash_centroid_density', 1
        #                'event_density', 2
        #                'average_flash_area', 3
        #                # 'standard_deviation_flash_area',
        #                'total_energy', 4
        #                'group_extent_density', 5
        #                'group_centroid_density', 6
        #                'average_group_area', 7
        #                'min_flash_area', 8
        #                )

        self.divide_grids[3]=0
        self.divide_grids[7]=5


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
    log.debug((n, s))
    dn = int(n/s)

    s_edges = np.arange(s+1, dtype='f8')*delta*dn + bnd[0]
    s_edges[-1] = bnd[1]
    return s_edges

def subdivided_fixed_grid(kwargs, process_flash_kwargs, out_kwargs, s=1,
    x_pad = 100*28.0e-6, y_pad = 100*28.0e-6):
    """

    Generator function to turn a single set of keyword arguments to
    grid_GLM_flashes into an s by s block of keyword arguments.

    x_pad and y_pad are padding in fixed grid coordinates used to increase the
    bounding box over which flashes are subset from the data file. A flash with
    a centroid near the edge of the target grid may have events within the
    grid, so we want to capture that flash too. Default is the equivalent of
    100 km at nadir.

    Yields (i,j) kwargs_ij, process_flash_kwargs_ij for each i,j subgrid.
    """
    # Convert padding into an integer multiple of dx
    n_x_pad = int(x_pad/kwargs['dx'])
    n_y_pad = int(y_pad/kwargs['dy'])
    x_pad = float(n_x_pad*kwargs['dx'])
    y_pad = float(n_y_pad*kwargs['dy'])

    pads = (n_x_pad, n_y_pad, x_pad, y_pad)

    x_sub_bnd = subdivide_bnd(kwargs['x_bnd'], kwargs['dx'], s=s)
    y_sub_bnd = subdivide_bnd(kwargs['y_bnd'], kwargs['dy'], s=s)
    log.debug((x_sub_bnd, y_sub_bnd))
    nadir_lon = process_flash_kwargs['nadir_lon']
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)

    for i, j in itertools.product(range(s), range(s)):
        kwargsij = kwargs.copy()
        prockwargsij = process_flash_kwargs.copy()
        x_bnd_i = x_sub_bnd[i:i+2].copy()
        y_bnd_j = y_sub_bnd[j:j+2].copy()
        # Make a copy of this and use it as target grid specs before
        # we modify to use as the flash selection bounding box.
        # These abut one other exactly.
        kwargsij['x_bnd'], kwargsij['y_bnd'] = x_bnd_i.copy(), y_bnd_j.copy()
        x_bnd_i[0] -= x_pad
        x_bnd_i[1] += x_pad
        y_bnd_j[0] -= y_pad
        y_bnd_j[1] += y_pad
        # This x_bnd and y_bnd are different from the one above in kwargsij.
        # They are used to subset the flashes, and therefore have additional
        # padding.
        prockwargsij['x_bnd'], prockwargsij['y_bnd'] = x_bnd_i, y_bnd_j

        # The line below guarantees overlap across the grids.
        # The flash selection bouning box and the target grid are the same
        kwargsij['x_bnd'], kwargsij['y_bnd'] = x_bnd_i, y_bnd_j

        # No need to do lon_bnd and lat_bnd because we subset in fixed grid
        # coordinates instead.
        prockwargsij['lon_bnd'], prockwargsij['lat_bnd'] = None, None

        outfile_prefix_base = out_kwargs['output_filename_prefix']
        outfile_prefix_base_ij = outfile_prefix_base + '-'
        outfile_prefix_base_ij += '{0:02d}-{1:02d}'.format(i,j)
        out_kwargs_ij = out_kwargs.copy()
        out_kwargs_ij['output_filename_prefix'] = outfile_prefix_base_ij

        # out_kwargs_ij['output_writer_cache'] = out_kwargs_ij['output_writer']
        preprocess_out = GridOutputPreprocess(
            pads=pads, writer=out_kwargs_ij['output_writer'])
        out_kwargs_ij['preprocess_out'] = preprocess_out
        out_kwargs_ij['output_writer'] = preprocess_out.capture_write_call

        log.debug(("SUBGRID", i,j, x_bnd_i, y_bnd_j, pads))

        yield (i, j), kwargsij, prockwargsij, out_kwargs_ij, pads

class GridOutputPreprocess(object):
    """
    The capture_write_call method of this class stands in for the call to
    output_writer in gridder.write_grids. It's used to trim the grids back to
    their actual shape, removing the padding added to ensure all flashes are
    captured on each grid.

    To actually write the data, call write_all, which uses the writer
    function passed on initialization.
    """
    def __init__(self, pads=None, writer=None):
        self.writer = writer
        self.pads = pads
        self.outargs=[]
        self.outkwargs=[]
    def capture_write_call(self, *args, **kwargs):
        # Use the padding information to trim up the grids
        log.info("Trimming grids")
        n_x_pad, n_y_pad, x_pad, y_pad = self.pads
        x_coord, y_coord = args[3], args[4]
        grid = args[9]
        if n_x_pad == 0:
            x_slice = slice(None, None)
        else:
            x_slice = slice(n_x_pad, -n_x_pad)
        if n_y_pad == 0:
            y_slice = slice(None, None)
        else:
            y_slice = slice(n_y_pad, -n_y_pad)
        args = (*args[:3], x_coord[x_slice], y_coord[y_slice], *args[5:9],
                grid[x_slice, y_slice], *args[10:])

        self.outargs.append(args)
        self.outkwargs.append(kwargs)
    def write_all(self):
        outfiles = []
        if self.writer:
            for outargs, outkwargs in zip(self.outargs, self.outkwargs):
                self.writer(*outargs, **outkwargs)
                outfiles.append(outargs[0])
        return outfiles


# @profile
def grid_GLM_flashes(GLM_filenames, start_time, end_time, **kwargs):
    """ Grid GLM data to a 2D grid between start_time and end_time.

    Arguments:
    start_time -- datetime object
    end_time -- datetime object

    Keyword arguments:
    proj_name -- string, one of [latlong, fixed_grid]. If latlong, the x_bnd and y_bnd
        kwargs are used as lon_bnd and lat_bnd.
    subdivide -- integer S, chop up the target grid into S x S tiles and process each
        tile in parallel (default 1). Only used for the fixed grid.

    Passed to GLMGridder.process_flashes:
        min_points_per_flash, min_groups_per_flash, clip_events,
        fixed_grid, nadir_lon, corner_pickle
    Passed to GLMGridder.write_grids:
        outpath, output_writer, output_writer_3d,
        output_kwargs, output_filename_prefix
    Remaining keyword arguments are passed to the GLMGridder on initialization.
    """

    kwargs['do_3d'] = False

    # Used only for the fixed grid at the moment
    subdivide_grid = kwargs.pop('subdivide', 1)

    process_flash_kwargs = {}
    for prock in ('min_points_per_flash','min_groups_per_flash',
                  'clip_events', 'fixed_grid', 'nadir_lon', 'corner_pickle',
                  'ellipse_rev'):
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
        pads = (0, 0, 0.0, 0.0)
        process_flash_kwargs['lon_bnd'] = kwargs['x_bnd']
        process_flash_kwargs['lat_bnd'] = kwargs['y_bnd']
        subgrids = [((0, 0), kwargs, process_flash_kwargs, out_kwargs, pads)]
    elif 'fixed_grid' in process_flash_kwargs:
        subgrids = subdivided_fixed_grid(kwargs, process_flash_kwargs,
                                         out_kwargs, s=subdivide_grid)
    else:
        # working with ccd pixels or a projection, so no known lat lon bnds
        process_flash_kwargs['lon_bnd'] = None
        process_flash_kwargs['lat_bnd'] = None
        pads = (0, 0, 0.0, 0.0)
        subgrids = [((0, 0), kwargs, process_flash_kwargs, out_kwargs, pads)]

    this_proc_each_grid = partial(proc_each_grid, start_time=start_time,
        end_time=end_time, GLM_filenames=GLM_filenames)

    if subdivide_grid > 1:
        pool = ProcessPoolExecutor(max_workers=4)
        with pool:
            # Block until the pool completes (pool is a context manager)
            outputs = pool.map(this_proc_each_grid, subgrids)
    else:
        outputs = list(map(this_proc_each_grid, subgrids))
    for op in outputs:
        log.debug(outputs)

    return outputs


# @profile
def proc_each_grid(subgrid, start_time=None, end_time=None, GLM_filenames=None):
    """ Process one tile (a subset of a larger grid) of GLM data.

    Arguments:
    subgrid -- tuple of (xi,yi), kwargs, proc_kwargs, out_kwargs, pads) where
        (xi, yi) -- the subgrid tile index
        kwargs -- passed to GLMGridder.__init__
        proc_kwargs -- passed to GLMGridder.process_flashes
        out_kwargs -- passed to GLMGridder.write_grids
        pads -- (n_x_pad, n_y_pad, x_pad, y_pad) counts and total distances of padding
            added to this subgrid

    Keyword arguments:
    start_time -- datetime object
    end_time -- datetime object
    GLM_filenames -- a list of GLM filenames to process
    """

    subgridij, kwargsij, process_flash_kwargs_ij, out_kwargs_ij, pads = subgrid 
    ellipse_rev = process_flash_kwargs_ij.pop('ellipse_rev')

    # Eventually, we want to trim off n_x/y_pad from each side of the grid
    n_x_pad, n_y_pad, x_pad, y_pad = pads

    log.info("out kwargs are", out_kwargs_ij)

    # These should all be independent at this point and can parallelize
    log.info(('gridder kwargs for subgrid {0} are'.format(subgridij), kwargsij))
    if 'clip_events' in process_flash_kwargs_ij:
        gridder = GLMlutGridder(start_time, end_time, **kwargsij)
    else:
        gridder = GLMGridder(start_time, end_time, **kwargsij)

    if 'clip_events' in process_flash_kwargs_ij:
        xedge,yedge=np.meshgrid(gridder.xedge,gridder.yedge)
        mesh = QuadMeshSubset(xedge, yedge, n_neighbors=16*10, regular=True)
        # import pickle
        # with open('/data/LCFA-production/L1b/mesh_subset.pickle', 'wb') as f:
            # pickle.dump(mesh, f)
        process_flash_kwargs_ij['clip_events'] = mesh
        log.debug(("XEDGE", subgridij, xedge))
        log.debug(("YEDGE", subgridij, yedge))
    for filename in GLM_filenames:
        # Could create a cache of GLM objects by filename here.
        log.info("Processing {0}".format(filename))
        log.info(('process flash kwargs for {0} are'.format(subgridij),
            process_flash_kwargs_ij))
        sys.stdout.flush()
        glm = GLMDataset(filename, ellipse_rev=ellipse_rev)
        # Pre-load the whole dataset, as recommended by the xarray docs.
        # This saves an absurd amount of time (factor of 80ish) in
        # grid.split_events.replicate_and_split_events
        if len(glm.dataset.number_of_events) > 0:
            # xarray 0.12.1 (and others?) throws an error when trying to load
            # data from an empty dimension.
            glm.dataset.load()
            gridder.process_flashes(glm, **process_flash_kwargs_ij)
        else:
            log.info("Skipping {0} - number of events is 0".format(filename))
        glm.dataset.close()
        del glm

    preprocess_out = out_kwargs_ij.pop('preprocess_out', None)
    if preprocess_out:
        output = gridder.write_grids(**out_kwargs_ij)
        outfilenames = preprocess_out.write_all()
    else:
        outfilenames = gridder.write_grids(**out_kwargs_ij)

    return (subgridij, outfilenames) # out_kwargs_ij
