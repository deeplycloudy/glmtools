import os, itertools
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

"""
SCALING of fields

---Flash extent density---
For 330 ms time separation critera, the max flash rate per minute in any pixel
is 181.8/min, or a max hourly value of 10860. 
10860/65535 = 1/6
Another way to go: for an 8x8 km GLM pixel oversampled to 2x2 km, 
the fractional area coverage is 1/16. This works out to 4095 flashes per minute
max, which seems plenty safe and covers the absolute max accumulation in 20 min.

2 bytes unsigned
scale=1/16=0.0625
offset=0

---Group extent density---
Have a max in relampago case of 1000/min (max is 30000/min for a 500 fps CCD),
so over 20 min this is 20000 events.
That is 5x the max we assume above in FED, which is equivalent to 5 groups/flash
assuming a full flash rate. And that full flash rate is unlikely to be reached!
4*1/16 = 1/4 would preserve a bit of antialiasing and multiplies nicely.
Set scale to 1/4 = 0.25
Offset = 0


---Flash and group centroid density---
GCD will be larger than FCD, so size with GCD. The max rate for CGD will likely
be the same as the GED - a maximum at the center of the cell. We can get away
with two bytes, scale and offset of 1 because there is no antialiasing.
scale=1
offset=0

---Average flash area, group area---
WMO record distance flash distance is 320 km, which squared is ~100,000 km^2.
Minimum observable flash area is 64 km^2, which divided by 16 (for smallest 
fractional pixel coverage at 2 km) is 4 km^2.
A 2 km^2 scale factor gets us 131070 max in two bytes.
However, since 10000 km^2 is the max flash area in the L2 data, it will never
be exceeded, and so we set scale to 1 and offset to 0, for a max 65535.

2 bytes, unsigned
scale_factor = 1 km^2
offset = 0

---Total optical energy---
2 bytes, linear
scale_factor=1.52597e-15 J, add_offset=0 (range 0 to 1.0e-10 J)

1.0e-10 is also the max event_energy value in the L2 files, so we can’t have
more than one event that hits that level. However, I think this should be safe.
The fact that the flash_energy has the same scale factor as event_energy in L2
suggests that there is margin in the ceiling of 1e-10 J. And as Scott's stats
showed, pixels with total energy in excess of even 10e-12 J are quite rare.
"""
glm_scaling = {
    'flash_extent_density':{'dtype':'uint16', 
        'scale_factor':0.0625, 'add_offset':0.0},
    'flash_centroid_density':{'dtype':'uint16', 
        'scale_factor':1.0, 'add_offset':0.0},
    'average_flash_area':{'dtype':'uint16', 
        'scale_factor':1.0, 'add_offset':0.0},
    'event_density':{'dtype':'uint16', 
        'scale_factor':0.25, 'add_offset':0.0},
    # 'standard_deviation_flash_area',
    'group_extent_density':{'dtype':'uint16', 
        'scale_factor':0.25, 'add_offset':0.0},
    'group_centroid_density':{'dtype':'uint16', 
        'scale_factor':1.0, 'add_offset':0.0},
    'average_group_area':{'dtype':'uint16', 
        'scale_factor':1.0, 'add_offset':0.0},
    'total_energy':{'dtype':'uint16', 
        'scale_factor':1.52597e-6, 'add_offset':0.0,},
}

def get_goes_imager_subpoint_vars(nadir_lon):
    """ Returns two xarray DataArrays containing the nominal satellite subpoint
        latitude and longitude, as netCDF float variables.
    
        returns subpoint_lon, subpoint_lat
    """
    sublat_meta = {}
    sublat_descr = "nominal satellite subpoint latitude (platform latitude)"
    sublat_meta['long_name'] = sublat_descr
    sublat_meta['standard_name'] = 'latitude'
    sublat_meta['units'] = 'degrees_north'
    sublat_enc = {'_FillValue':-999.0}
    
    sublon_meta = {}
    sublon_descr = "nominal satellite subpoint longitude (platform longitude)"
    sublon_meta['long_name'] = sublon_descr
    sublon_meta['standard_name'] = 'longitude'
    sublon_meta['units'] = 'degrees_east'
    sublon_enc = {'_FillValue':-999.0}
            
    sublat = xr.DataArray(0.0, name='nominal_satellite_subpoint_lat',
                          attrs=sublat_meta)
    sublat.encoding = sublat_enc
    sublon = xr.DataArray(nadir_lon, name='nominal_satellite_subpoint_lon',
                          attrs=sublon_meta)
    sublon.encoding = sublon_enc
    return sublon, sublat

def get_goes_imager_proj(nadir_lon):
    """ Returns an xarray DataArray containing the GOES-R series
        goes_imager_projection data and metadata
    """
    meta = {}
    meta['long_name'] = "GOES-R ABI fixed grid projection"
    meta['grid_mapping_name'] = "geostationary"
    meta['perspective_point_height'] = 35786023.
    meta['semi_major_axis'] = 6378137.
    meta['semi_minor_axis'] = 6356752.31414
    meta['inverse_flattening'] = 298.2572221
    meta['latitude_of_projection_origin'] = 0.0
    meta['longitude_of_projection_origin'] = nadir_lon
    meta['sweep_angle_axis'] = "x"
    
    encoding = {}
    encoding['dtype'] = 'i4'
    
    var = xr.DataArray(-2147483647, attrs=meta, name='goes_imager_projection')
    var.encoding=encoding
    return var

def get_goes_imager_all_valid_dqf(dims, n):
    """ dims is a tuple of dimension names in the same order as n, the number
        of elements along each dimension
    
    Returns dqf, an xarray.DataArray of the GLM data quality
        field, in the style of the GOES-R series DQF field """
    
    meta = {}
    meta['grid_mapping'] = "goes_imager_projection"
    meta['number_of_qf_values'] = np.asarray(6, dtype='i4')
    meta['units'] = "1"
    meta['standard_name'] = "status_flag"
    meta['long_name'] = "GLM data quality flags"
    meta['flag_values'] = np.asarray((0,1), dtype='i4')
    meta['flag_meanings'] = "valid, invalid"
    
    dqf = np.zeros(n, dtype='u1')
    
    enc = {}
    enc['_FillValue'] = np.asarray(255, dtype='u1')
    enc['_Unsigned'] = "true"
    enc['dtype'] = 'i1'
    enc['zlib'] = True # compress the field
    
    dqf_var = xr.DataArray(dqf, dims=dims, attrs=meta, name="DQF")
    dqf_var.encoding = enc
    return dqf_var
    
def get_goes_imager_fixedgrid_coords(x, y, resolution='2km at nadir',
        scene_id='FULL', fill=-999.0):
    """ Create variables with metadata for fixed grid coordinates as defined
    for the GOES-R series of spacecraft.

    Assumes that imagery are at 2 km resolution (no other options are
    implemented), and applies the scale and offset values indicated in the
    GOES-R PUG for the full disk scene, guaranteeing that we cover all fixed
    grid coordinates.
    
    Arguments:
    x, y: 1-dimensional arrays of coordinate values
    resolution: like "2km at nadir"
    scene_id: 'FULL' is the only allowed argument; other values will be ignored 

    Returns: 
    x_var, y_var: xarray.DataArray objects, with type inferred from x and y.
    """
    scene_id='FULL'
    # Values from the GOES-R PUG. These are signed shorts (int16).
    two_km_enc = {
        'FULL':{'dtype':'int16', 'x':{'scale_factor': 0.000056,
                                      'add_offset':-0.151844,
                                      '_FillValue':-999.0},
                                 'y':{'scale_factor':-0.000056,
                                      'add_offset':0.151844,
                                      '_FillValue':-999.0},
               },
    # The PUG has specific values for the CONUS sector, and
    # given the discretization of the coords to 2 km resolution, is it necessary
    # to special-case each scene so the span of the image? Right now ONLY
    # GOES-EAST CONUS IS IMPLEMENTED as a special case, with scene_id='CONUS'.

        # 'CONUS':{'dtype':'int16', 'x':{'scale_factor': 0.000056,
        #                               'add_offset':-0.101332},
        #                          'y':{'scale_factor':-0.000056,
        #                               'add_offset':0.128212},
        #        }
        # 'MESO1', 'MESO2', 'OTHER'
        }
    # two_km_enc['OTHER'] = two_km_enc['MESO1']
    
    x_meta, y_meta = {}, {}
    x_enc = two_km_enc['FULL']['x']
    x_enc['dtype'] = two_km_enc[scene_id]['dtype']
    y_enc = two_km_enc['FULL']['y']
    y_enc['dtype'] = two_km_enc[scene_id]['dtype']

    
    x_meta['axis'] = "X"
    x_meta['long_name'] = "GOES fixed grid projection x-coordinate"
    x_meta['standard_name'] = 'projection_x_coordinate'
    x_meta['units'] = "rad"

    y_meta['axis'] = "Y"
    y_meta['long_name'] = "GOES fixed grid projection y-coordinate"
    y_meta['standard_name'] = 'projection_y_coordinate'    
    y_meta['units'] = "rad"
    
    x_coord = xr.DataArray(x, name='x', dims=('x',),
                           attrs=x_meta)
    x_coord.encoding = x_enc
    y_coord = xr.DataArray(y, name='y', dims=('y',),
                           attrs=y_meta)
    y_coord.encoding = y_enc
    return x_coord, y_coord


def get_glm_global_attrs(start, end, platform, slot, instrument, scene_id,
        resolution, timeline, prod_env, prod_src, prod_site, ):
    """
    Create the global metadata attribute dictionary for GOES-R series GLM
    Imagery products.

    Arguments:
    start, end: datetime of the start and end times of image coverage
    platform: one of G16, G17 or a follow-on platform
    slot: the orbital slot ("GOES-East", "GOES-West", etc.)
    instrument: one of "GLM-1", "GLM-2", or a follow on instrument.
    scene_id: one of 'FULL', 'CONUS', 'MESO1', or 'MESO2' if compatible with
        the ABI definitions or 'OTHER'.
    resolution: like "2km at nadir" 
    prod_env: "OE", "DE", etc.
    prod_src: "Realtime" or "Postprocessed"
    prod_site: "NAPO", "TTU", etc.

    The date_created is set to the time at which this function is run. 
    
    Returns: meta, a dictionary of metadata attributes.
    """
    created = datetime.now()
    modes = {'ABI Mode 3':'M3'}
    
    # For use in the dataset name / filename
    scenes = {'FULL':'F',
              'CONUS':'C',
              'MESO1':'M1',
              'MESO2':'M2',
              'OTHER':'M1'}
    scene_names = {"FULL":"Full Disk",
                   "CONUS":"CONUS",
                   "MESO1":"Mesoscale",
                   "MESO2":"Mesoscale",
                   "OTHER":"Custom"}

    # "OR_GLM-L2-GLMC-M3_G16_s20181011100000_e20181011101000_c20181011124580.nc 
    dataset_name = "OR_GLM-L2-GLM{5}-{0}_{1}_s{2}_e{3}_c{4}.nc".format(
        modes[timeline], platform, start.strftime('%Y%j%H%M%S0'),
        end.strftime('%Y%j%H%M%S0'), created.strftime('%Y%j%H%M%S0'),
        scenes[scene_id]
    )
    
    meta = {}

    # Properties that don't change
    meta['cdm_data_type'] = "Image" 
    meta['Conventions'] = "CF-1.7" 
    meta['id'] = "93cb84a3-31ef-4823-89f5-c09d88fc89e8" 
    meta['institution'] = "DOC/NOAA/NESDIS > U.S. Department of Commerce, National Oceanic and Atmospheric Administration, National Environmental Satellite, Data, and Information Services" 
    meta['instrument_type'] = "GOES R Series Geostationary Lightning Mapper" 
    meta['iso_series_metadata_id'] = "f5816f53-fd6d-11e3-a3ac-0800200c9a66" 
    meta['keywords'] = "ATMOSPHERE > ATMOSPHERIC ELECTRICITY > LIGHTNING, ATMOSPHERE > ATMOSPHERIC PHENOMENA > LIGHTNING" 
    meta['keywords_vocabulary'] = "NASA Global Change Master Directory (GCMD) Earth Science Keywords, Version 7.0.0.0.0" 
    meta['license'] = "Unclassified data.  Access is restricted to approved users only." 
    meta['Metadata_Conventions'] = "Unidata Dataset Discovery v1.0" 
    meta['naming_authority'] = "gov.nesdis.noaa" 
    meta['processing_level'] = "National Aeronautics and Space Administration (NASA) L2" 
    meta['project'] = "GOES"
    meta['standard_name_vocabulary'] = "CF Standard Name Table (v25, 05 July 2013)" 
    meta['summary'] = "The Lightning Detection Gridded product generates fields starting from the GLM Lightning Detection Events, Groups, Flashes product.  It consists of flash extent density, event density, average flash area, average group area, total energy, flash centroid density, and group centroid density." 
    meta['title'] = "GLM L2 Lightning Detection Gridded Product"

    # Properties that change
    meta['dataset_name'] = dataset_name
    meta['date_created'] = created.isoformat()+'Z'
    meta['instrument_ID'] = instrument
    meta['orbital_slot'] = slot
    meta['platform_ID'] = platform
    meta['production_data_source'] = prod_src
    meta['production_environment'] = prod_env
    meta['production_site'] = prod_site 
    meta['scene_id'] = scene_names[scene_id]
    meta['spatial_resolution'] = resolution
    meta['time_coverage_end'] = end.isoformat()+'Z'
    meta['time_coverage_start'] = start.isoformat()+'Z'
    meta['timeline_id'] = timeline
    
    return meta
    
def glm_image_to_var(data, name, long_name, units, dims, fill=0.0,
                     scale_factor=None, add_offset=None, dtype=None):
    """
    data: array of data
    name: the standard name, CF-compliant if possible
    long_name: a more descriptive name
    units: udunits string for the units of data
    dims: tuple of coordinate names

    dtype: numpy dtype of variable to be written after applying scale and offset
    
    If dtype is not None, then the following are also checked
    scale_factor, add_offset: floating point discretization and offset, as
        commonly used in NetCDF datasets.
        decoded = scale_factor * encoded + add_offset
    
    Returns: data_var, xarray.DataArray objects, with type inferred from data

    """
    enc = {}
    meta = {}
    
    enc['_FillValue'] = fill
    enc['zlib'] = True # Compress the data
    if dtype is not None:
        orig_dtype = dtype
        if orig_dtype[0] == 'u':
            enc['_Unsigned'] = 'true'
            dtype= dtype[1:]
        enc['dtype'] = dtype
        if scale_factor is not None:
            enc['scale_factor'] = scale_factor
        if add_offset is not None:
            enc['add_offset'] = add_offset
    meta['standard_name'] = name
    meta['long_name'] = long_name
    meta['units'] = units
    meta['grid_mapping'] = "goes_imager_projection"
    
    d = xr.DataArray(data, attrs=meta, dims=dims, name=name)
    d.encoding = enc
    return d

def new_goes_imagery_dataset(x, y, nadir_lon):
    """ Create a new xarray.Dataset with the basic coordiante data, metadata,
    and global attributes that matches the GOES-R series fixed grid imagery
    format.
    
    Arguments
    x, y: 1-D arrays of coordinate positions
    nadir_lon: longitude (deg) of the sub-satellite point
    """

    # Dimensions
    dims = ('y', 'x')
    
    scene_id, nominal_resolution = infer_scene_from_dataset(x, y)
    log.debug("Span of grid implies scene is {0}".format(scene_id))

    # Coordinate data: x, y
    xc, yc = get_goes_imager_fixedgrid_coords(x, y, scene_id=scene_id)
    # Coordinate reference system
    goes_imager_proj = get_goes_imager_proj(nadir_lon)
    subpoint_lon, subpoint_lat = get_goes_imager_subpoint_vars(nadir_lon)
    
    # Data quality flags
    dqf = get_goes_imager_all_valid_dqf(dims, y.shape+x.shape)
    
    v = {goes_imager_proj.name:goes_imager_proj,
         dqf.name:dqf,
         subpoint_lat.name:subpoint_lat,
         subpoint_lon.name:subpoint_lon,
        }
    c = {xc.name:xc, yc.name:yc}
    d = xr.Dataset(data_vars=v, coords=c)#, dims=dims)
    # Attributes aren't carried over for xc and yc like they are for dqf, etc.
    # so copy them over manually
    d.x.attrs.update(xc.attrs)
    d.y.attrs.update(yc.attrs)
    
    return d, scene_id, nominal_resolution

def xy_to_2D_lonlat(gridder, x_coord, y_coord):
    self = gridder
    
    mapProj = self.mapProj
    geoProj = self.geoProj

    x_all, y_all = (a.T for a in np.meshgrid(x_coord, y_coord))
    assert x_all.shape == y_all.shape
    assert x_all.shape[0] == nx
    assert x_all.shape[1] == ny
    z_all = np.zeros_like(x_all)

    lons, lats, alts = x,y,z = geoProj.fromECEF( 
            *mapProj.toECEF(x_all, y_all, z_all) )
    lons.shape=x_all.shape
    lats.shape=y_all.shape
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def infer_scene_from_dataset(x, y):
    "Infer whether the scene matches one of the GOES-R fixed grid domains."
    from lmatools.grid.fixed import goesr_conus, goesr_meso, goesr_full, goesr_resolutions
    rtol = 1.0e-2

    # Try to match up the actual spacing in microradians with a known resolutions
    dx = np.abs(x[1]-x[0])
    resolution = '{:d}microradian at nadir'.format(int(np.round(dx*1e6)))
    for km, microrad in goesr_resolutions.items():
        if np.allclose(microrad, dx, rtol=rtol):
            resolution = km.replace('.0', '') + ' at nadir'

    spanEW = x.max() - x.min()
    spanNS = y.max() - y.min()
    log.debug("Inferring scene from spans x={0}, y={1}".format(spanEW, spanNS))
    if   (np.allclose(spanEW, goesr_full['spanEW'], rtol=rtol) &
          np.allclose(spanNS, goesr_full['spanNS'], rtol=rtol) ):
        scene_id = "FULL"
    elif (np.allclose(spanEW, goesr_conus['spanEW'], rtol=rtol) &
          np.allclose(spanNS, goesr_conus['spanNS'], rtol=rtol) ):
        scene_id = "CONUS"
    elif (np.allclose(spanEW, goesr_meso['spanEW'], rtol=rtol) &
          np.allclose(spanNS, goesr_meso['spanNS'], rtol=rtol) ):
        scene_id = "MESO1"
    else:
        scene_id = "OTHER"
    return scene_id, resolution

def write_goes_imagery(gridder, outpath='.', pad=None, scale_and_offset=True):
    """ pad is a tuple of x_slice, y_slice: slice objects used to index the
            zeroth and first dimensions, respectively, of the grids in gridder.
    
        scale_and_offset controls whether to write variables as scaled ints.
        if False, floating point grids will be written.
        
    """
                # output_filename_prefix="LMA", **output_kwargs):
    self = gridder
    if pad is not None:
        x_slice, y_slice = pad
    else:
        x_slice, y_slice = (slice(None, None), slice(None, None))
    
    fixProj = self.mapProj
    geoProj = self.geoProj
    
    # Center of the grid is the nadir longitude
    subsat = (0.0, 0.0, 0.0)
    nadir_lon, nadir_lat, nadir_alt = geoProj.fromECEF(*fixProj.toECEF(*subsat))
        
    # Get 1D x and y coordinates, and corresponding lons and lats
    spatial_scale_factor = self.spatial_scale_factor
    xedge = self.xedge
    yedge = self.yedge
    
    x_coord = ((xedge[:-1] + xedge[1:])/2.0)[x_slice]
    y_coord = ((yedge[:-1] + yedge[1:])/2.0)[y_slice]
    
    file_iter = list(zip(self.outgrids, self.field_names,
                 self.field_descriptions, self.field_units, self.outformats))
    
    # Write a separate file at each time.
    all_outfiles = []
    for ti, (t0, t1) in enumerate(pairwise(self.t_edges_seconds)):
        start = self.t_ref + timedelta(0, t0)
        end = self.t_ref + timedelta(0, t1)
        log.info("Assembling NetCDF dataset for {0} - {1}".format(start, end))
        
        # Need to flip the y coordinate to reverse order since y is defined as
        # upper left in the GOES-R series L1b PUG (section 5.1.2.6 Product Data
        # Structures). Later, to follow the image array convention will
        # transpose the grids and then flipud.
        dataset, scene_id, nominal_resolution = new_goes_imagery_dataset(x_coord,
                                        np.flipud(y_coord), nadir_lon)

        # Global metadata
        l2lcfa_attrs = gridder.first_file_attrs
        
        global_attrs = get_glm_global_attrs(start, end,
                l2lcfa_attrs['platform_ID'], l2lcfa_attrs['orbital_slot'],
                l2lcfa_attrs['instrument_ID'], scene_id,
                nominal_resolution, "ABI Mode 3", "DE", "Postprocessed", "TTU"
                )
        dataset = dataset.assign_attrs(**global_attrs)
        # log.debug("*** Checking x coordinate attrs initial")
        # log.debug(dataset.x.attrs)
                
        outfile = os.path.join(outpath, dataset.attrs['dataset_name'])

        # Adding a new variable to the dataset below clears the coord attrs
        # so hold on to them for now.
        xattrs, yattrs = dataset.x.attrs, dataset.y.attrs
        xenc, yenc = dataset.x.encoding, dataset.y.encoding
        
        for i, (grid_allt, field_name, description, units, outformat) in enumerate(file_iter):
            grid = grid_allt[x_slice,y_slice,ti]
            if i in self.divide_grids:
                denom = self.outgrids[self.divide_grids[i]][x_slice,y_slice,ti]
                zeros = (denom == 0) | (grid == 0)
                nonzeros = ~zeros
                grid[nonzeros] = grid[nonzeros]/denom[nonzeros]
                grid[zeros] = 0 # avoid nans
            image_at_time = np.flipud(grid.T)

            scale_kwargs = {}
            if (field_name in glm_scaling) and scale_and_offset:
                scale_kwargs.update(glm_scaling[field_name])
            img_var = glm_image_to_var(image_at_time,
                                       field_name, description, units,
                                       ('y', 'x'), **scale_kwargs)
            # Why does this line clear the attrs on the coords?
            # log.debug("*** Checking x coordinate attrs {0}a".format(i))
            # log.debug(dataset.x.attrs)
            dataset[img_var.name] = img_var
            # log.debug("*** Checking x coordinate attrs {0}b".format(i))
            # log.debug(dataset.x.attrs)

        # Restore the cleared coord attrs
        dataset.x.attrs.update(xattrs)
        dataset.y.attrs.update(yattrs)
        dataset.x.encoding.update(xenc)
        dataset.y.encoding.update(yenc)

        # log.debug("*** Checking x coordinate attrs final")
        # log.debug(dataset.x.attrs)
            
        log.info("Preparing to write NetCDF {0}".format(outfile))
        dataset.to_netcdf(outfile)
        log.info("Wrote NetCDF {0}".format(outfile))
        all_outfiles.append(outfile)
    return all_outfiles


def aggregate(glm, minutes, start_end=None):
    """ Given a multi-minute glm imagery dataset (such as that returned by
        glmtools.io.imagery.open_glm_time_series) and an integer number of minutes,
        recalculate average and minimum flash area for that interval and sum all other
        fields.
    
        start_end: datetime objects giving the start and end edges of the interval to
            be aggregated. This allows for a day-long dataset to be aggregated over an hour
            of interest, for example. If not provided, the start of the glm dataset plus
            *minutes* after the end of the glm dataset will be used.
    
        To restore the original time coordinate name, choose the left, mid, or right
        endpoint of the time_bins coordinate produced by the aggregation step.
        >>> agglm = aggregate(glm, 5)
        >>> agglm['time_bins'] = [v.left for v in agglm.time_bins.values]
        >>> glm_agg = agglm.rename({'time_bins':'time'})
    """
    dt_1min = timedelta(seconds=60)
    dt = dt_1min*minutes
    
    if start_end is not None:
        start = start_end[0]
        end = start_end[1]
    else:
        start = pd.Timestamp(glm['time'].min().data).to_pydatetime()
        end = pd.Timestamp(glm['time'].max().data).to_pydatetime() + dt
        # dt_np = (end - start).data
        # duration = pd.to_timedelta(dt_np).to_pytimedelta()
    duration = end - start
    
    sum_vars = ['flash_extent_density', 'flash_centroid_density',
                'total_energy',
                'group_extent_density', 'group_centroid_density', ]
    sum_vars = [sv for sv in sum_vars if sv in glm]
    sum_data = glm[sum_vars]
                    
    # goes_imager_projection is a dummy int variable, and all we care about
    # is the attributes.
    min_vars = ['minimum_flash_area', 'goes_imager_projection',
                    'nominal_satellite_subpoint_lat', 'nominal_satellite_subpoint_lon']
    min_vars = [mv for mv in min_vars if mv in glm]
    min_data = glm[min_vars]
    
    

    sum_data['total_flash_area'] = glm.average_flash_area*glm.flash_extent_density
    sum_data['total_group_area'] = glm.average_flash_area*glm.flash_extent_density

    t_bins = [start + dt*i for i in range(int(duration/dt)+1)]
    t_groups_sum = sum_data.groupby_bins('time', bins=t_bins)
    t_groups_min = min_data.groupby_bins('time', bins=t_bins)

    # Take the minimum of anything along the minimum dimension. We need to account for
    # zero values, however, so that we don't cancel out pixels where there is a flash in
    # one minute but not the other. TODO TODO TODO
    aggregated_min = t_groups_min.min(dim='time', keep_attrs=True)    
    
    # Naively sum all variables … so average areas are now ill defined. Recalculate
    aggregated = t_groups_sum.sum(dim='time', keep_attrs=True)
    aggregated['average_flash_area'] = (aggregated.total_flash_area
                                        / aggregated.flash_extent_density)
    aggregated['average_group_area'] = (aggregated.total_group_area
                                        / aggregated.group_extent_density)
    for var in min_vars:
        aggregated[var] = aggregated_min[var]
    
    # direct copy of other useful attributes. This could be handled better, since it brings 
    # along the original time dimension, but the projection is a static value and is safe
    # to move over. We skip 'DQF' since it's empty and not clear what summing it would mean.
    # Someone can make that decision later when DQF is populated.
    # for v in ['goes_imager_projection']:
        # aggregated[v] = glm[v]
        
    # time_bins is made up of Interval objects with left and right edges
    aggregated.attrs['time_coverage_start'] = min(
        [v.left for v in aggregated.time_bins.values]).isoformat()
    aggregated.attrs['time_coverage_end'] = max(
        [v.right for v in aggregated.time_bins.values]).isoformat()
    
    return aggregated



def gen_file_times(filenames, time_attr='time_coverage_start'):
    for s in filenames:
        with xr.open_dataset(s) as d:
            # strip off timezone information so that xarray
            # auto-converts to datetime64 instead of pandas.Timestamp objects
            yield pd.Timestamp(d.attrs[time_attr]).tz_localize(None)


def open_glm_time_series(filenames, chunks=None):
    """ Convenience function for combining individual 1-min GLM gridded imagery
    files into a single xarray.Dataset with a time dimension.
    
    Creates an index on the time dimension.
    
    The time dimension will be in the order in which the files are listed
    due to the behavior of combine='nested' in open_mfdataset.
    
    Adjusts the time_coverage_start and time_coverage_end metadata.
    """
    # Need to fix time_coverage_start and _end in concat dataset
    starts = [t for t in gen_file_times(filenames)]
    ends = [t for t in gen_file_times(filenames, time_attr='time_coverage_end')]
    
    d = xr.open_mfdataset(filenames, concat_dim='time', chunks=chunks, combine='nested')
    d['time'] = starts
    d = d.set_index({'time':'time'})
    d = d.set_coords('time')
    
    d.attrs['time_coverage_start'] = pd.Timestamp(min(starts)).isoformat()
    d.attrs['time_coverage_end'] = pd.Timestamp(max(ends)).isoformat()

    return d