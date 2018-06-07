import numpy as np
import xarray as xr

import pkg_resources

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# from concurrent.futures import ProcessPoolExecutor
from functools import partial

from glmtools.io.glm import GLMDataset, get_lutevents
from glmtools.grid.split_events import split_event_data, split_event_dataset_from_props, split_group_dataset_from_props, split_flash_dataset_from_props, replicate_and_weight_split_child_dataset
from glmtools.grid.clipping import QuadMeshPolySlicer, join_polys
from glmtools.io.ccd import load_pixel_corner_lookup, quads_from_corner_lookup
from glmtools.io.lightning_ellipse import ltg_ellps_lon_lat_to_fixed_grid
from lmatools.io.LMA_h5_file import LMAh5Collection
from lmatools.lasso.cell_lasso_timeseries import TimeSeriesGenericFlashSubset
from lmatools.lasso.energy_stats import TimeSeriesPolygonLassoFilter
from lmatools.grid.fixed import get_GOESR_coordsys

# from lmatools.io.LMA_h5_file import LMAh5Collection

# h5LMAfiles=('/data/20130606/LMA/LYLOUT_130606_032000_0600.dat.flash.h5',)

# h5s = LMAh5Collection(h5LMAfiles, base_date=panels.basedate, min_points=10)
# for events, flashes in h5s: # get a stream of events, flashes
#     print events.dtype
#     print flashes.dtype

def read_flashes(glm, target, base_date=None, lon_range=None, lat_range=None,
                 x_range=None, y_range=None,
                 min_events=None, min_groups=None, clip_events=True,
                 fixed_grid=False, nadir_lon=None, chunk_size=20000,
                 corner_pickle=None):
    """ This routine is the data pipeline source, responsible for pushing out
        events and flashes. Using a different flash data source format is a
        matter of replacing this routine to read in the necessary event
        and flash data.

        Arguments:
        glm -- a glmtools.io.glm.GLMDataset instance
        target -- dictionary, with keys 'flash', 'group', and 'event' pointing to the
            pipeline inlet for each, to which (child, parent) arrays are sent for each.
            If None, return a dictionary with the keys above, and values (child, parent)
            for each.

        Keyword arguments:
        base_date -- when calculating time in seconds of the day, use the date given
            by this datetime object.
        lon_range, lat_range -- subset flashes to some portion of the full disk.
            (Default None to use all data.)
        x_range, y_range -- as with lon_range, lat_range but for coordinates on
            the ABI fixed grid. (Default None to use all data.)
        min_events, min_groups -- subset only those flashes with at least this many
            events or groups. (Default None to use all data.)
        clip_events -- Use the corner point lookup table (see corner_pickle) to
            reconstruct the event, group, and flash polygons and split them by the target
            grid.
        fixed_grid -- Convert GLM L2 lon, lat to ABI fixed grid coordinates
        nadir_lon -- longitude of the satellite subpoint
        chunk_size -- process this many flashes at one time (default 100)
        corner_pickle -- Path to a pickled look up table giving the corner point offsets
            for events as a function of location. If None, use the built-in fixed grid
            corner point look up table.

        See the documentation for read_flash_chunk for how the above arguments are used.
     """
    if corner_pickle is None:
        resource_package = __name__  # Could be any module/package name
        resource_path = '/'.join(('G16_corner_lut_fixedgrid.pickle',))
        corner_pickle = pkg_resources.resource_filename(resource_package, resource_path)
        # print(corner_pickle)
    if ((lon_range is not None) | (lat_range is not None) |
        (x_range is not None) | (y_range is not None) |
        (min_events is not None) | (min_groups is not None)):
        # only subset if we have to
        log.info("Subsetting flashes")
        flash_data = glm.subset_flashes(
                        x_range=x_range, y_range=y_range,
                        lon_range=lon_range, lat_range=lat_range,
                        min_events=min_events, min_groups=min_groups)
    else:
        flash_data = glm.dataset

    flash_ids = flash_data.flash_id.data[:]
    n_chunks = max(int(flash_ids.shape[0]/chunk_size),1)
    flash_chunks = []
    for ichunk, id_chunk in enumerate(np.array_split(flash_ids, n_chunks)):
        log.info("Grabbing chunk {0} of {2} for file {1}".format(ichunk+1, glm._filename, n_chunks))
        flash_chunks.append(glm.get_flashes(id_chunk))

    if clip_events:
        chunk_func = partial(fast_fixed_grid_read_chunk, target=target, base_date=base_date,
                             nadir_lon=nadir_lon, clip_events=clip_events,
                             corner_pickle=corner_pickle)
    else:
        chunk_func = partial(read_flash_chunk, glm=glm, target=target, base_date=base_date,
                             nadir_lon=nadir_lon, fixed_grid=fixed_grid,
                             clip_events=clip_events,
                             corner_pickle=corner_pickle)

    # # This doesn't work because we can't pickle the targets - they're generators
    # pool = ProcessPoolExecutor(max_workers=4)
    # with pool:
    #     # Block until the pool completes (pool is a context manager)
    #     results = list(pool.map(chunk_func, flash_chunks))

    results = list(map(chunk_func,flash_chunks))

    return results

def fast_fixed_grid_read_chunk(flash_data, target=None, base_date=None, nadir_lon=None,
         clip_events=True, corner_pickle=None):
    """ For the chunk of GLM data in flash_data, prepare a dataset that can be
    sent to lmatools for accumulation on a target grid.

    Assumes that we're running in the full-quality mode, using both clip_events
    and fixed_grid. ``glmtools.io.glm.get_lutevents`` is called on flash_data to
    pre-aggregate the group and flash properties to a discretized event
    location.
    Thereafter, the following steps take place:
    1. GLM event polygons are reconstructed from a corner point lookup table.
    2. The GLM event polygons are split by the target grid
    3. For each of the split polygons, calculate its centroid, area, fraction of
         the original polygon area, and fraction of the target grid cell covered
    4. For each split polygon, replicate and weight the pre-accumulated
         flash, group, and event data
    5. Create numpy arrays with named dtype (i.e., data table with named columns) for
         (child, parent) pairs at the event, group, and flash levels. See the
         documentation for mimic_lma_dataset_lut.
    6. Send these data to the target, or return them directly if target is None.

    Arguments:
    flash_data -- xarray dataset giving the flash data, like that produced by a call
         to GLMDataset.subset_flashes followed by a call to get_lutevents.

    Keyword arguments:
    target -- dictionary, with keys 'flash', 'group', and 'event' pointing to the
        pipeline inlet for each, to which (child, parent) arrays are sent for each.
        If None, return a dictionary with the keys above, and values (child, parent)
        for each.
    base_date -- when calculating time in seconds of the day, use the date given
        by this datetime object.
    nadir_lon -- longitude of the satellite subpoint.
    clip_events -- An instance of glmtools.grid.clipping.QuadMeshSubset
        used to slice the GLM event polygons by the target grid given
        by QuadMesh. The corner point lookup table (see corner_pickle) is used to
        reconstruct the event polygons.
    corner_pickle -- Path to a pickled look up table giving the corner point offsets
        for events as a function of location. If None, use the built-in fixed grid
        corner point look up table.
    """
    flash_data = get_lutevents(flash_data)
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)

    split_event_dataset=None
    split_group_dataset=None
    split_flash_dataset=None

    mesh = clip_events
    event_ids = flash_data.lutevent_id.data
    event_x, event_y = flash_data.lutevent_x, flash_data.lutevent_y

    x_lut, y_lut, corner_lut = load_pixel_corner_lookup(corner_pickle)
    # Convert from microradians to radians
    x_lut = x_lut * 1.0e-6
    y_lut = y_lut * 1.0e-6
    corner_lut = corner_lut*1e-6
    event_polys = quads_from_corner_lookup(x_lut, y_lut, corner_lut,
        event_x, event_y)
    # event_polys_inflated = event_polys
    # event_polys_inflated = quads_from_corner_lookup(x_lut, y_lut,
    #     corner_lut, event_x, event_y, inflate=1.02)

    slicer = QuadMeshPolySlicer(mesh)
    # --- Split up the events ---
    log.info("Starting to split events")
    chopped_polys, poly_areas = slicer.slice(event_polys)
    log.info("Sliced all events")
    split_event_polys, split_event_properties = split_event_data(
        chopped_polys, poly_areas, slicer, event_ids=event_ids)

    # Use variable names that indicate the fixed grid polygon centroid
    # coordinates are x,y. Then, convert those coordinates to lon, lat
    # referenced to the earth (not lightning) ellipsoid, because
    # lmatools expects lon, lat as input to its gridding routines.
    # These will be wastefully re-transformed back into fixed grid
    # coordinates when gridded onto the fixed grid, but this approch
    # also allows gridding to arbitrary other target grids.
    split_event_dataset = split_event_dataset_from_props(
        split_event_properties, centroid_names=('split_event_x',
                                                'split_event_y'))
    fixed_x_ctr = split_event_dataset['split_event_x'].data
    fixed_y_ctr = split_event_dataset['split_event_y'].data
    fixed_z_ctr = np.zeros_like(fixed_x_ctr)
    split_event_lon, split_event_lat, split_event_alt = grs80lla.fromECEF(
        *geofixcs.toECEF(fixed_x_ctr, fixed_y_ctr, fixed_z_ctr))
    split_dims = getattr(split_event_dataset,
        'split_event_parent_event_id').dims
    split_event_dataset['split_event_lon'] = (split_dims, split_event_lon)
    split_event_dataset['split_event_lat'] = (split_dims, split_event_lat)

    log.info("Calculated split event properties")
    split_event_dataset = replicate_and_weight_split_child_dataset_new(
        flash_data, split_event_dataset)
    log.info("Finished replicated event dataset")

    try:
        fake_lma = mimic_lma_dataset_lut(flash_data, base_date,
            split_events=split_event_dataset)
        # *_flashes and *_events and are really the properties of the
        # parent entity (one row per entity) and their constituent
        # components that cover space. Don't take "_events" and
        # "_flashes" literally here.
        # The terminology is inherited from the LMA infrastructure
        # where flashes are made up of consituent points, so that there
        # are only two levels of hierarchy, and where the events have no
        # spatial extent that can be further split.
        fl_events, fl_flashes = fake_lma['flash']
        gr_events, gr_flashes = fake_lma['group']
        ev_events, ev_flashes = fake_lma['event']

        if target is not None:
            if fl_events.shape[0] >= 1:
                flash_target = target['flash']
                group_target = target['group']
                event_target = target['event']
                flash_target.send((fl_events, fl_flashes))
                group_target.send((gr_events, gr_flashes))
                event_target.send((ev_events, ev_flashes))
                del fl_events, fl_flashes, ev_events, ev_flashes
                del gr_events, gr_flashes, fake_lma
        else:
            return fake_lma
    except KeyError as ke:
        err_txt = 'Skipping {0}\n    ... assuming a flash, group, or event with id {1} does not exist'
        log.error(err_txt.format(glm.dataset.dataset_name, ke))


def replicate_and_weight_split_child_dataset_new(parent_data, split_child_dataset,
        parent_id='lutevent_id', split_child_parent_id='split_event_parent_event_id',
        names=['lutevent_id', 'lutevent_energy', 'lutevent_time_offset',
               'lutevent_count', 'lutevent_flash_count', 'lutevent_group_count',
               'lutevent_total_flash_area', 'lutevent_total_group_area'],
        weights={'lutevent_energy':'split_event_area_fraction',
                 'lutevent_count': 'split_event_mesh_area_fraction',
                 'lutevent_flash_count': 'split_event_mesh_area_fraction',
                 'lutevent_group_count': 'split_event_mesh_area_fraction',
                 'lutevent_total_flash_area':'split_event_mesh_area_fraction',
                 'lutevent_total_group_area':'split_event_mesh_area_fraction',
                }
        ):
    """
    Replicate variables from parents to their children. The children in this case
    correspond to some fraction of the total area covered by parents, i.e., the children
    are the centroids of polygons resulting from subdivision of a parent polygon.
    Optionally, weight each parent variable by some variable included in the split
    child data.

    The default arguments show how to accumulate and weight event data that have been
    pre-aggregated using an event lookup table approach (glmtools.io.glm.get_lutevents).

    Arguments:
    parent_data -- xarray dataset, containing data variables for the parent
    split_child_dataset -- xarray dataset, for children resulting from splitting the
        parent polygon. Includes centroids and fractional coverage of the parent.

    Keyword arguemnts:
    parent_id -- variable name in parent_data corresponding to replicated parent_ids in
        split_child_parent_event_id within split_child_dataset
    split_child_parent_event_id -- variable name in split_child_dataset (foreign key)
        corresponding to parent_id in parent_data
    names -- the variables in parent_data to replicate and weight
    weights -- dictionary with key giving a variable in parent_data and value giving the
        variable in split_child_dataset to use for weighting that variable. Only the
        names in weights will be weighted.

    For data such as parent energy, it is typical to weight with the fraction of the
    original parent's area covered by the split child, so that the total energy is
    conserved. This is usually named 'split_event_area_fraction' or similar.

    For count data, such as that used for calculating extent density, it is typical to
    weight with the fraction of the coverage of the target grid cell. Imagine a grid cell
    covered 40 percent by one parent (with a count of "3") and another 60 percent by an
    adjacent parent, also with a count of "3". The desired result is that the target grid
    cell have a final count of three. This is usually named something like
    'split_event_mesh_area_fraction'.
    """

    split_dims = getattr(split_child_dataset, split_child_parent_id).dims

    # We have a bunch of split events, but the original dataset only has data
    # from the parent once for each event. Need to replicate those data by
    # using the split_child_parent_id to index the original dataset.
    replicated_event_ids = getattr(split_child_dataset, split_child_parent_id)

    replicated_data = parent_data.loc[{parent_id:replicated_event_ids}]

    for name in names:
        new_name = 'split_' + name
        new_var = getattr(replicated_data, name).data
        if name in weights:
            weight_var = getattr(split_child_dataset, weights[name])
            # dimension names won't match, but lengths should.
            new_var = new_var*weight_var.data # should ensure copy, not view
        # add variable to the dataset
        split_child_dataset[new_name] = (split_dims, new_var) #{'dims':split_dims, 'data':new_var}

    return split_child_dataset


def read_flash_chunk(flash_data, glm=None, target=None, base_date=None, nadir_lon=None,
         fixed_grid=False, clip_events=True, corner_pickle=None):
    """ For the chunk of GLM data in flash_data, prepare a dataset that can be sent
    to lmatools for accumulation on a target grid.

    In the full-quality mode (using both clip_events and fixed_grid), the following
    steps take place:
    0. GLM data are renavigated to the GLM fixed grid and the lightning ellipse removed.
    1. GLM event polygons are reconstructed from a corner point lookup table.
    2. The events for each group (flash) are joined into polygons for each group (flash).
    3. Split the event polygons by the target grid
    4. Split the group polygons by the target grid
    5. Split the flash polygons by the target grid
    6. For each of the split polygons, calculate its centroid, area, fraction of
         the original polygon area, and fraction of the target grid cell covered
    7. Create numpy arrays with named dtype (i.e., data table with named columns) for
         (child, parent) pairs at the event, group, and flash levels. See the
         documentation for mimic_lma_dataset.
    8. Send these data to the target, or return them directly if target is None.

    Arguments:
    flash_data -- xarray dataset giving the flash data, like that produced by a call
         to glm.subset_flashes

    Keyword arguments:
    glm -- a glmtools.io.glm.GLMDataset instance
    target -- dictionary, with keys 'flash', 'group', and 'event' pointing to the
        pipeline inlet for each, to which (child, parent) arrays are sent for each.
        If None, return a dictionary with the keys above, and values (child, parent)
        for each.
    base_date -- when calculating time in seconds of the day, use the date given
        by this datetime object.
    nadir_lon -- longitude of the satellite subpoint.
    fixed_grid -- Convert GLM L2 lon, lat to ABI fixed grid coordinates
    clip_events -- If an instance of glmtools.grid.clipping.QuadMeshSubset is provided,
        use it to slice the GLM event, group, and flash polygons by the target grid given
        by QuadMesh. The corner point lookup table (see corner_pickle) is used to
        reconstruct the event, group, and flash polygons.
    corner_pickle -- Path to a pickled look up table giving the corner point offsets
        for events as a function of location. If None, use the built-in fixed grid
        corner point look up table.
    """
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)

    split_event_dataset=None
    split_group_dataset=None
    split_flash_dataset=None
    if clip_events:
        mesh = clip_events
        event_lons = flash_data.event_lon.data
        event_lats = flash_data.event_lat.data
        event_ids = flash_data.event_id.data
        event_parent_flash_ids = flash_data.event_parent_flash_id.data
        event_parent_group_ids = flash_data.event_parent_group_id.data

        # On a per-flash basis, and eventually on a per-group basis,
        # we want to union all events for each parent. This gives a master
        # polygon for each parent which we then split into individual sub-polys
        # defined by slicing with the target grid.

        # For now, we ignore the group level entirely, and reduce the
        # flash down to sub-polys, which we treat as lma-like events.

        # For flashes, and in particular the flash extent density product,
        # it is necessary that he entire flash footprint be a single polygon
        # with no slivers of missed coverage on the interior.
        # When sliced by the grid, the slivers result in more than one
        # polygon for each target grid cell. The calculation of flash extent
        # density uses the fractional area of the original grid cell covered by
        # these polygons, but only one (unpredictably chosen) sub-polygon for
        # each mesh cell is used. Therefore, we inflate the event_polys before
        # unioning by a little bit to avoid the above problem.

        if fixed_grid:
            x_lut, y_lut, corner_lut = load_pixel_corner_lookup(corner_pickle)
            # Convert from microradians to radians
            x_lut = x_lut * 1.0e-6
            y_lut = y_lut * 1.0e-6
            corner_lut = corner_lut*1e-6
            event_x, event_y = ltg_ellps_lon_lat_to_fixed_grid(event_lons,
                event_lats, nadir_lon)
            event_polys = quads_from_corner_lookup(x_lut, y_lut, corner_lut,
                event_x, event_y)
            event_polys_inflated = quads_from_corner_lookup(x_lut, y_lut,
                corner_lut, event_x, event_y, inflate=1.02)
        else:
            lon_lut, lat_lut, corner_lut = load_pixel_corner_lookup(
                corner_pickle)
                # '/data/LCFA-production/L1b/G16_corner_lut_lonlat.pickle')
            event_polys = quads_from_corner_lookup(lon_lut, lat_lut,
                corner_lut, event_lons, event_lats, nadir_lon=nadir_lon)
            event_polys_inflated = quads_from_corner_lookup(lon_lut, lat_lut,
                corner_lut, event_lons, event_lats, nadir_lon=nadir_lon,
                inflate=1.02)

        slicer = QuadMeshPolySlicer(mesh)
        # --- Split up the events ---
        log.info("Starting to split events")
        chopped_polys, poly_areas = slicer.slice(event_polys)
        log.info("Sliced all events")
        split_event_polys, split_event_properties = split_event_data(chopped_polys, poly_areas, slicer, event_ids=event_ids)
        if fixed_grid:
            # Use variable names that indicate the fixed grid polygon centroid
            # coordinates are x,y. Then, convert those coordinates to lon, lat
            # referenced to the earth (not lightning) ellipsoid, because
            # lmatools expects lon, lat as input to its gridding routines.
            # These will be wastefully re-transformed back into fixed grid
            # coordinates when gridded onto the fixed grid, but this approch
            # also allows gridding to arbitrary other target grids.
            split_event_dataset = split_event_dataset_from_props(
                split_event_properties, centroid_names=('split_event_x',
                                                        'split_event_y'))
            fixed_x_ctr = split_event_dataset['split_event_x'].data
            fixed_y_ctr = split_event_dataset['split_event_y'].data
            fixed_z_ctr = np.zeros_like(fixed_x_ctr)
            split_event_lon, split_event_lat, split_event_alt = grs80lla.fromECEF(
                *geofixcs.toECEF(fixed_x_ctr, fixed_y_ctr, fixed_z_ctr))
            split_dims = getattr(split_event_dataset,
                'split_event_parent_event_id').dims
            split_event_dataset['split_event_lon'] = (split_dims, split_event_lon)
            split_event_dataset['split_event_lat'] = (split_dims, split_event_lat)
        else:
            split_event_dataset = split_event_dataset_from_props(split_event_properties)
        log.info("Calculated split event properties")
        split_event_dataset = replicate_and_weight_split_child_dataset(glm, split_event_dataset)
        log.info("Finished replicated event dataset")
        # --- Now split the groups ---
        log.info("Starting to split groups")
        unique_gr_ids = np.unique(event_parent_group_ids)
        poly_index = dict(((k,[]) for k in unique_gr_ids))
        for split_poly, group_id in zip(event_polys_inflated, event_parent_group_ids):
            poly_index[group_id].append(split_poly)

        union_polys = []
        union_group_ids = []
        for gr_id, gr_polys in poly_index.items():
            gr_join_poly = join_polys(gr_polys)
            union_polys.extend(gr_join_poly)
            union_group_ids.extend([gr_id,]*len(gr_join_poly))

        chopped_union_polys, poly_union_areas = slicer.slice(union_polys, bbox=True)
        log.info("Sliced all groups")
        split_group_polys, split_group_properties = split_event_data(chopped_union_polys,
            poly_union_areas, slicer, event_ids=union_group_ids)
        if fixed_grid:
            # Use variable names that indicate the fixed grid polygon centroid
            # coordinates are x,y. Then, convert those coordinates to lon, lat
            # referenced to the earth (not lightning) ellipsoid, because
            # lmatools expects lon, lat as input to its gridding routines.
            # These will be wastefully re-transformed back into fixed grid
            # coordinates when gridded onto the fixed grid, but this approch
            # also allows gridding to arbitrary other target grids.
            split_group_dataset = split_group_dataset_from_props(
                split_group_properties, centroid_names=('split_group_x',
                                                        'split_group_y'))
            fixed_x_ctr = split_group_dataset['split_group_x'].data
            fixed_y_ctr = split_group_dataset['split_group_y'].data
            fixed_z_ctr = np.zeros_like(fixed_x_ctr)
            split_group_lon, split_group_lat, split_group_alt = grs80lla.fromECEF(
                *geofixcs.toECEF(fixed_x_ctr, fixed_y_ctr, fixed_z_ctr))
            split_dims = getattr(split_group_dataset,
                'split_group_parent_group_id').dims
            split_group_dataset['split_group_lon'] = (split_dims, split_group_lon)
            split_group_dataset['split_group_lat'] = (split_dims, split_group_lat)
        else:
            split_group_dataset = split_group_dataset_from_props(
                split_group_properties)
        log.info("Calculated split group properties")
        split_group_dataset = replicate_and_weight_split_child_dataset(glm, split_group_dataset,
            parent_id='group_id', split_child_parent_id='split_group_parent_group_id',
            names=['group_time_offset', 'group_area', 'group_energy'],
            weights={'group_energy':'split_group_area_fraction',
                     'group_area':'split_group_area_fraction'})
        log.info("Finished replicated group dataset")
        # --- Now split the flashes ---
        log.info("Starting to split flashes")
        unique_fl_ids = np.unique(event_parent_flash_ids)
        poly_index = dict(((k,[]) for k in unique_fl_ids))
        for split_poly, flash_id in zip(event_polys_inflated, event_parent_flash_ids):
            poly_index[flash_id].append(split_poly)

        union_polys = []
        union_flash_ids = []
        for fl_id, fl_polys in poly_index.items():
            fl_join_poly = join_polys(fl_polys)
            union_polys.extend(fl_join_poly)
            union_flash_ids.extend([fl_id,]*len(fl_join_poly))
        # all_flash_sub_polys = [p for p in itertools.chain.from_iterable(poly_index.values())]

        chopped_union_polys, poly_union_areas = slicer.slice(union_polys, bbox=True)
        log.info("Sliced all flashes")
        split_flash_polys, split_flash_properties = split_event_data(chopped_union_polys,
            poly_union_areas, slicer, event_ids=union_flash_ids)
        if fixed_grid:
            # Use variable names that indicate the fixed grid polygon centroid
            # coordinates are x,y. Then, convert those coordinates to lon, lat
            # referenced to the earth (not lightning) ellipsoid, because
            # lmatools expects lon, lat as input to its gridding routines.
            # These will be wastefully re-transformed back into fixed grid
            # coordinates when gridded onto the fixed grid, but this approch
            # also allows gridding to arbitrary other target grids.
            split_flash_dataset = split_flash_dataset_from_props(
                split_flash_properties, centroid_names=('split_flash_x',
                                                        'split_flash_y'))
            fixed_x_ctr = split_flash_dataset['split_flash_x'].data
            fixed_y_ctr = split_flash_dataset['split_flash_y'].data
            fixed_z_ctr = np.zeros_like(fixed_x_ctr)
            split_flash_lon, split_flash_lat, split_flash_alt = grs80lla.fromECEF(
                *geofixcs.toECEF(fixed_x_ctr, fixed_y_ctr, fixed_z_ctr))
            split_dims = getattr(split_flash_dataset,
                'split_flash_parent_flash_id').dims
            split_flash_dataset['split_flash_lon'] = (split_dims, split_flash_lon)
            split_flash_dataset['split_flash_lat'] = (split_dims, split_flash_lat)
        else:
            split_flash_dataset = split_flash_dataset_from_props(
                split_flash_properties)
        log.info("Calculated split flash properties")
        split_flash_dataset = replicate_and_weight_split_child_dataset(glm, split_flash_dataset,
            parent_id='flash_id', split_child_parent_id='split_flash_parent_flash_id',
            names=['flash_time_offset_of_first_event', 'flash_time_offset_of_last_event',
                   'flash_area', 'flash_energy'],
            weights={'flash_energy':'split_flash_area_fraction',
                     'flash_area':'split_flash_area_fraction'})
        log.info("Finished replicated flash dataset")

    try:
        fake_lma = mimic_lma_dataset(flash_data, base_date,
            split_events=split_event_dataset,
            split_groups=split_group_dataset,
            split_flashes=split_flash_dataset)
        # *_flashes and *_events and are really the properties of the
        # parent entity (one row per entity) and their constituent
        # components that cover space. Don't take "_events" and
        # "_flashes" literally here.
        # The terminology is inherited from the LMA infrastructure
        # where flashes are made up of consituent points, so that there
        # are only two levels of hierarchy, and where the events have no
        # spatial extent that can be further split.
        fl_events, fl_flashes = fake_lma['flash']
        gr_events, gr_flashes = fake_lma['group']
        ev_events, ev_flashes = fake_lma['event']

        if target is not None:
            if fl_events.shape[0] >= 1:
                flash_target = target['flash']
                group_target = target['group']
                event_target = target['event']
                flash_target.send((fl_events, fl_flashes))
                group_target.send((gr_events, gr_flashes))
                event_target.send((ev_events, ev_flashes))
                del fl_events, fl_flashes, ev_events, ev_flashes
                del gr_events, gr_flashes, fake_lma
        else:
            return fake_lma
    except KeyError as ke:
        err_txt = 'Skipping {0}\n    ... assuming a flash, group, or event with id {1} does not exist'
        log.error(err_txt.format(glm.dataset.dataset_name, ke))


def sec_since_basedate(t64, basedate):
    """ given a numpy datetime 64 object, and a datetime basedate,
        return seconds since basedate"""

    t_series = t64 - np.datetime64(basedate)
    t = np.divide(t_series, np.timedelta64(1, 's'))
    return t


# These are the dtypes in the LMA HDF5 data files
event_dtype=[('flash_id', '<i4'),
             ('alt', '<f4'),
#                  ('charge', 'i1'), ('chi2', '<f4'), ('mask', 'S4'), ('stations', 'u1'),
             ('lat', '<f4'), ('lon', '<f4'), ('time', '<f8'),
             ('mesh_frac', '<f8'),
             ('mesh_xi', '<i4'), ('mesh_yi', '<i4'),
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

def _fake_lma_from_glm_flashes(flash_data, basedate,
        split_events=None, split_groups=None, split_flashes=None):
    """ Helper function used by mimic_lma_dataset.

    `flash_data` is an xarray dataset of flashes, groups, and events for
    (possibly more than one) lightning flash. `flash_data` can be generated
    with `GLMDataset.subset_flashes` or `GLMDataset.get_flashes`.

    Here we create fake LMA events using the flash-level data. If we have
    split flash data, it means the fake LMA events were generated by
    splitting the polygon resulting from the union of all events that are a
    part of each flash.
    """

    flash_np = np.empty_like(flash_data.flash_id.data, dtype=flash_dtype)
    if split_events is not None:
        event_np = np.empty_like(split_flashes.split_flash_lon.data, dtype=event_dtype)
    else:
        event_np = np.empty_like(flash_data.event_id.data, dtype=event_dtype)

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return event_np, flash_np

    if split_flashes is not None:
        event_np['flash_id'] = split_flashes.split_flash_parent_flash_id.data
        event_np['lat'] = split_flashes.split_flash_lat
        event_np['lon'] = split_flashes.split_flash_lon
        t_event = sec_since_basedate(split_flashes.split_flash_time_offset_of_first_event.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = split_flashes.split_flash_energy
        event_np['mesh_frac'] = np.abs(split_flashes.split_flash_mesh_area_fraction.data)
        event_np['mesh_xi'] = split_flashes.split_flash_mesh_x_idx.data
        event_np['mesh_yi'] = split_flashes.split_flash_mesh_y_idx.data
    else:
        event_np['flash_id'] = flash_data.event_parent_flash_id.data
        event_np['lat'] = flash_data.event_lat
        event_np['lon'] = flash_data.event_lon
        t_event = sec_since_basedate(flash_data.event_time_offset.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = flash_data.event_energy

    flash_np['area'] = flash_data.flash_area.data
    flash_np['total_energy'] = flash_data.flash_energy.data
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

def _fake_lma_from_glm_groups(flash_data, basedate,
        split_events=None, split_groups=None, split_flashes=None):
    """ Helper function used by mimic_lma_dataset.

    `flash_data` is an xarray dataset of flashes, groups, and events for
    (possibly more than one) lightning flash. `flash_data` can be generated
    with `GLMDataset.subset_flashes` or `GLMDataset.get_flashes`.

    Here we create fake LMA events using the group-level data. If we have
    split group data, it means the fake LMA events were generated by
    splitting the polygon resulting from the union of all events that are a
    part of each group.
    """

    flash_np = np.empty_like(flash_data.group_id.data, dtype=flash_dtype)
    if split_events is not None:
        event_np = np.empty_like(split_groups.split_group_lon.data, dtype=event_dtype)
    else:
        event_np = np.empty_like(flash_data.event_id.data, dtype=event_dtype)

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return event_np, flash_np

    if split_groups is not None:
        event_np['flash_id'] = split_groups.split_group_parent_group_id.data
        event_np['lat'] = split_groups.split_group_lat
        event_np['lon'] = split_groups.split_group_lon
        t_event = sec_since_basedate(split_groups.split_group_time_offset.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = split_groups.split_group_energy
        event_np['mesh_frac'] = np.abs(split_groups.split_group_mesh_area_fraction.data)
        event_np['mesh_xi'] = split_groups.split_group_mesh_x_idx.data
        event_np['mesh_yi'] = split_groups.split_group_mesh_y_idx.data
    else:
        event_np['flash_id'] = flash_data.event_parent_group_id.data
        event_np['lat'] = flash_data.event_lat
        event_np['lon'] = flash_data.event_lon
        t_event = sec_since_basedate(flash_data.event_time_offset.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = flash_data.event_energy

    flash_np['area'] = flash_data.group_area.data
    flash_np['total_energy'] = flash_data.group_energy.data
    flash_np['ctr_lon'] = flash_data.group_lon.data
    flash_np['ctr_lat'] = flash_data.group_lat.data
    flash_np['init_lon'] = flash_data.group_lon.data
    flash_np['init_lat'] = flash_data.group_lat.data
    t_start = sec_since_basedate(flash_data.group_time_offset.data, basedate)
    t_end = sec_since_basedate(flash_data.group_time_offset.data, basedate)
    flash_np['start'] = t_start
    flash_np['duration'] = t_end-t_start
    flash_np['flash_id'] = flash_data.group_id.data
    flash_np['n_points'] = flash_data.number_of_events.shape[0]

    # Fake the altitude data
    event_np['alt'] = 0.0
    flash_np['ctr_alt'] = 0.0
    flash_np['init_alt'] = 0.0

    # Fake the specific energy data
    flash_np['specific_energy'] = 0.0

    return event_np, flash_np


def _fake_lma_from_glm_events(flash_data, basedate,
        split_events=None, split_groups=None, split_flashes=None):
    """ Helper function used by mimic_lma_dataset

    `flash_data` is an xarray dataset of flashes, groups, and events for
    (possibly more than one) lightning flash. `flash_data` can be generated
    with `GLMDataset.subset_flashes` or `GLMDataset.get_flashes`.

    Here we create fake LMA events using the event-level data. If we have
    split event data, it means the fake LMA events were generated by
    splitting each event polygon across the target grid. So, in this case
    the flash_np data are actually the GLM event polygons.
    """

    flash_np = np.empty_like(flash_data.event_id.data, dtype=flash_dtype)
    if split_events is not None:
        event_np = np.empty_like(split_events.split_event_lon.data, dtype=event_dtype)
    else:
        event_np = np.empty_like(flash_data.event_id.data, dtype=event_dtype)

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return event_np, flash_np

    if split_events is not None:
        event_np['flash_id'] = split_events.split_event_parent_event_id.data
        event_np['lat'] = split_events.split_event_lat
        event_np['lon'] = split_events.split_event_lon
        t_event = sec_since_basedate(split_events.split_event_time_offset.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = split_events.split_event_energy
        event_np['mesh_frac'] = np.abs(split_events.split_event_mesh_area_fraction.data)
        event_np['mesh_xi'] = split_events.split_event_mesh_x_idx.data
        event_np['mesh_yi'] = split_events.split_event_mesh_y_idx.data
    else:
        event_np['flash_id'] = flash_data.event_id.data
        event_np['lat'] = flash_data.event_lat.data
        event_np['lon'] = flash_data.event_lon.data
        t_event = sec_since_basedate(flash_data.event_time_offset.data, basedate)
        event_np['time'] = t_event
        event_np['power'] = flash_data.event_energy.data

    flash_np['area'] = 0.0 # flash_data.event_area.data - not in the dataset
    flash_np['total_energy'] = flash_data.event_energy.data
    flash_np['ctr_lon'] = flash_data.event_lon.data
    flash_np['ctr_lat'] = flash_data.event_lat.data
    flash_np['init_lon'] = flash_data.event_lon.data
    flash_np['init_lat'] = flash_data.event_lat.data
    t_start = sec_since_basedate(flash_data.event_time_offset.data, basedate)
    t_end = sec_since_basedate(flash_data.event_time_offset.data, basedate)
    flash_np['start'] = t_start
    flash_np['duration'] = t_end-t_start
    flash_np['flash_id'] = flash_data.event_id.data
    flash_np['n_points'] = flash_data.number_of_events.shape[0]

    # Fake the altitude data
    event_np['alt'] = 0.0
    flash_np['ctr_alt'] = 0.0
    flash_np['init_alt'] = 0.0

    # Fake the specific energy data
    flash_np['specific_energy'] = 0.0

    return event_np, flash_np


def _fake_lma_events_from_split_glm_lutevents(split_events, basedate):
    lut_split_event_dtype=[('flash_id', '<i4'),
                 ('alt', '<f4'),
                 ('lat', '<f4'), ('lon', '<f4'), ('time', '<f8'),
                 ('mesh_frac', '<f8'),
                 ('mesh_xi', '<i4'), ('mesh_yi', '<i4'),
                 ('power', '<f4'),
                 ('lutevent_count', 'f4'),
                 ('lutevent_flash_count', 'f4'),
                 ('lutevent_group_count', 'f4'),
                 ('lutevent_total_flash_area', 'f4'),
                 ('lutevent_total_group_area', 'f4'),
                 ]

    event_np = np.empty_like(split_events.split_event_lon.data,
        dtype=lut_split_event_dtype)

    if event_np.shape[0] == 0:
        # no data, nothing to do
        return event_np

    event_np['flash_id'] = split_events.split_event_parent_event_id.data
    event_np['lat'] = split_events.split_event_lat
    event_np['lon'] = split_events.split_event_lon
    event_np['alt'] = 0.0
    t_event = sec_since_basedate(split_events.split_lutevent_time_offset.data, basedate)
    event_np['time'] = t_event
    event_np['power'] = split_events.split_lutevent_energy.data
    event_np['mesh_frac'] = np.abs(split_events.split_event_mesh_area_fraction.data)
    event_np['mesh_xi'] = split_events.split_event_mesh_x_idx.data
    event_np['mesh_yi'] = split_events.split_event_mesh_y_idx.data
    event_np['lutevent_count'] = split_events.split_lutevent_count.data
    event_np['lutevent_flash_count'] = split_events.split_lutevent_flash_count.data
    event_np['lutevent_group_count'] = split_events.split_lutevent_group_count.data
    event_np['lutevent_total_flash_area'] = split_events.split_lutevent_total_flash_area.data
    event_np['lutevent_total_group_area'] = split_events.split_lutevent_total_group_area.data

    return event_np


def _fake_lma_from_glm_lutflashes(flash_data, basedate):
    flash_np = np.empty_like(flash_data.flash_id.data, dtype=flash_dtype)

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return flash_np

    flash_np['area'] = flash_data.flash_area.data
    flash_np['total_energy'] = flash_data.flash_energy.data
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
    flash_np['ctr_alt'] = 0.0
    flash_np['init_alt'] = 0.0

    # Fake the specific energy data
    flash_np['specific_energy'] = 0.0

    return flash_np

def _fake_lma_from_glm_lutgroups(flash_data, basedate):
    flash_np = np.empty_like(flash_data.group_id.data, dtype=flash_dtype)

    if flash_np.shape[0] == 0:
        # no data, nothing to do
        return flash_np

    flash_np['area'] = flash_data.group_area.data
    flash_np['total_energy'] = flash_data.group_energy.data
    flash_np['ctr_lon'] = flash_data.group_lon.data
    flash_np['ctr_lat'] = flash_data.group_lat.data
    flash_np['init_lon'] = flash_data.group_lon.data
    flash_np['init_lat'] = flash_data.group_lat.data
    t_start = sec_since_basedate(flash_data.group_time_offset.data, basedate)
    t_end = sec_since_basedate(flash_data.group_time_offset.data, basedate)
    flash_np['start'] = t_start
    flash_np['duration'] = t_end-t_start
    flash_np['flash_id'] = flash_data.group_id.data
    flash_np['n_points'] = flash_data.number_of_events.shape[0]

    # Fake the altitude data
    flash_np['ctr_alt'] = 0.0
    flash_np['init_alt'] = 0.0

    # Fake the specific energy data
    flash_np['specific_energy'] = 0.0

    return flash_np


def mimic_lma_dataset_lut(flash_data, basedate, split_events=None):
    """ Mimic the LMA data structure from GLM data. Copies data to the data structure
    and variable names expected by the flash accumulation pipeline.

    Arguments:
    flash_data -- xarray dataset of GLM data.
    basedate -- Use the date given by this datetime object when calculating time in
        seconds of the day as needed by lmatools

    Keyword arguments:
    split_events -- pre-aggregated, split events that have been
        accumulated using an event lookup table before splitting and weighting

    Returns:
    fake_lma -- dictionary with keys 'flash', 'group', and 'event'. Each key has a
        value (child, parent). These arrays mimic the format lmatools expects, and
        are treated as point data when used by by lmatools. The child points
        ("events" in lmatools) are the sub-polygons, while the parent points
        ("flashes" in lmatools) are the properties of the parent event, group, or
        flash. There will be one or more child points for each parent.
    """
    ev_events = _fake_lma_events_from_split_glm_lutevents(split_events, basedate)
    fl_events = gr_events = ev_events

    ev_flashes = ev_events
    gr_flashes = _fake_lma_from_glm_lutgroups(flash_data, basedate)
    fl_flashes = _fake_lma_from_glm_lutflashes(flash_data, basedate)

    fake_lma = {'flash': (fl_events, fl_flashes),
                'group': (gr_events, gr_flashes),
                'event': (ev_events, ev_flashes),
               }
    return fake_lma


def mimic_lma_dataset(flash_data, basedate,
                      split_events=None,
                      split_groups=None,
                      split_flashes=None):
    """ Mimic the LMA data structure from GLM data.

    Arguments:
    flash_data -- xarray dataset of GLM data.
    basedate -- Use the date given by this datetime object when calculating time in
        seconds of the day as needed by lmatools

    Returns:
    fake_lma -- dictionary with keys 'flash', 'group', and 'event'. Each key has a
        value (child, parent). These arrays mimic the format lmatools expects, and
        are treated as point data when used by by lmatools. The child points
        ("events" in lmatools) are the sub-polygons, while the parent points
        ("flashes" in lmatools) are the properties of the parent event, group, or
        flash. There will be one or more child points for each parent.
    """

    fl_events, fl_flashes = _fake_lma_from_glm_flashes(flash_data, basedate,
        split_events=split_events, split_groups=split_groups,
        split_flashes=split_flashes)
    gr_events, gr_flashes = _fake_lma_from_glm_groups(flash_data, basedate,
        split_events=split_events, split_groups=split_groups,
        split_flashes=split_flashes)
    ev_events, ev_flashes = _fake_lma_from_glm_events(flash_data, basedate,
        split_events=split_events, split_groups=split_groups,
        split_flashes=split_flashes)

    fake_lma = {'flash': (fl_events, fl_flashes),
                'group': (gr_events, gr_flashes),
                'event': (ev_events, ev_flashes),
               }
    return fake_lma



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
        fake_lma = read_flashes(glm, None, base_date=self.base_date,
                                min_events=self.min_points,
                                min_groups=self.min_groups,
                                lon_range=self.lon_range,
                                lat_range=self.lat_range,
                                clip_events=False)
        ev_chunks = (d['flash'][0] for d in fake_lma)
        fl_chunks = (d['flash'][0] for d in fake_lma)
        events, flashes = np.hstack(ev_chunks), np.hstack(fl_chunks)
        log.info('data from {0}'.format(fname))
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

