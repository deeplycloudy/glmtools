import os, itertools
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

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
    
    var = xr.DataArray(-2147483647, attrs=meta, name='goes_imager_projection')
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
    
    dqf_var = xr.DataArray(dqf, dims=dims, attrs=meta, encoding=enc, name="DQF")
    
    return dqf_var
    
def get_goes_imager_fixedgrid_coords(x, y):
    """ Create variables with metadata for fixed grid coordinates as defined
    for the GOES-R series of spacecraft.
    
    Arguments:
    x, y: 1-dimensional arrays of coordinate values
    
    Returns: 
    x_var, y_var: xarray.DataArray objects, with type inferred from x and y.
    """
    x_meta, y_meta = {}, {}

    x_meta['axis'] = "X"
    x_meta['long_name'] = "GOES fixed grid projection x-coordinate"
    x_meta['standard_name'] = 'projection_x_coordinate'
    x_meta['units'] = "rad"

    y_meta['axis'] = "Y"
    y_meta['long_name'] = "GOES fixed grid projection y-coordinate"
    y_meta['standard_name'] = 'projection_y_coordinate'    
    y_meta['units'] = "rad"
    
    x_coord = xr.DataArray(x, attrs=x_meta, name='x', dims=('x',))
    y_coord = xr.DataArray(y, attrs=y_meta, name='y', dims=('y',))
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
    scene_id: one of 'FULL', 'CONUS', 'MESO' if compatible with the ABI definitions or 'none'
    resolution: like "2km at nadir" 
    prod_env: "OE", "DE", etc.
    prod_src: "Realtime" or "Postprocessed"
    prod_site: "NAPO", "TTU", etc.

    The date_created is set to the time at which this function is run. 
    
    Returns: meta, a dictionary of metadata attributes.
    """
    created = datetime.now()
    modes = {'ABI Mode 3':'M3'}
    
    # "OR_GLM-L2-GLMC-M3_G16_s20181011100000_e20181011101000_c20181011124580.nc 
    dataset_name = "OR_GLM-L2-GLMC-{0}_{1}_s{2}_e{3}_c{4}.nc".format(
        modes[timeline], platform, start.strftime('%Y%j%H%M%S0'),
        end.strftime('%Y%j%H%M%S0'), created.strftime('%Y%j%H%M%S0')
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
    meta['scene_id'] = scene_id
    meta['spatial_resolution'] = resolution
    meta['time_coverage_end'] = end.isoformat()+'Z'
    meta['time_coverage_start'] = start.isoformat()+'Z'
    meta['timeline_id'] = timeline
    
    return meta
    
def glm_image_to_var(data, name, long_name, units, dims, fill=0.0):
    """
    data: array of data
    name: the standard name, CF-compliant if possible
    long_name: a more descriptive name
    units: udunits string for the units of data
    dims: tuple of coordinate names
    
    Returns: data_var, xarray.DataArray objects, with type inferred from data

    """
    enc = {}
    meta = {}
    
    enc['_FillValue'] = fill
    enc['zlib'] = True # Compress the data
    meta['standard_name'] = name
    meta['long_name'] = long_name
    meta['units'] = units
    meta['grid_mapping'] = "goes_imager_projection"
    
    d = xr.DataArray(data, attrs=meta, encoding=enc, dims=dims, name=name)
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
    
    # Coordinate data: x, y
    xc, yc = get_goes_imager_fixedgrid_coords(x, y)
    # Coordinate reference system
    goes_imager_proj = get_goes_imager_proj(nadir_lon)
    
    # Data quality flags
    dqf = get_goes_imager_all_valid_dqf(dims, y.shape+x.shape)
    
    v = {goes_imager_proj.name:goes_imager_proj,
         dqf.name:dqf
        }
    c = {xc.name:xc, yc.name:yc}
    d = xr.Dataset(data_vars=v, coords=c)#, dims=dims)
    # Attributes aren't carried over for xc and yc like they are for dqf, etc.
    # so copy them over manually
    d.x.attrs.update(xc.attrs)
    d.y.attrs.update(yc.attrs)
    
    return d

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

def write_goes_imagery(gridder, outpath='.', pad=None):
    """ pad is a tuple of x_slice, y_slice: slice objects used to index the
            zeroth and first dimensions, respectively, of the grids in gridder.
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
    for ti, (t0, t1) in enumerate(pairwise(self.t_edges_seconds)):
        start = self.t_ref + timedelta(0, t0)
        end = self.t_ref + timedelta(0, t1)
        log.info("Assembling NetCDF dataset for {0} - {1}".format(start, end))
        
        # Need to flip the y coordinate to reverse order since y is defined as
        # upper left in the GOES-R series L1b PUG (section 5.1.2.6 Product Data
        # Structures). Later, to follow the image array convention will
        # transpose the grids and then flipud.
        dataset = new_goes_imagery_dataset(x_coord, np.flipud(y_coord),
                                           nadir_lon)
        # Global metadata
        global_attrs = get_glm_global_attrs(start, end,
                "G16", "GOES-East", "GLM-1", "Custom",
                "2km at nadir", "ABI Mode 3", "DE", "Postprocessed", "TTU"
                )
        dataset = dataset.assign_attrs(**global_attrs)
        # log.debug("*** Checking x coordinate attrs initial")
        # log.debug(dataset.x.attrs)
                
        outfile = os.path.join(outpath, dataset.attrs['dataset_name'])

        # Adding a new variable to the dataset below clears the coord attrs
        # so hold on to them for now.
        xattrs, yattrs = dataset.x.attrs, dataset.y.attrs
        
        for i, (grid_allt, field_name, description, units, outformat) in enumerate(file_iter):
            grid = grid_allt[x_slice,y_slice,ti]
            if i in self.divide_grids:
                denom = self.outgrids[self.divide_grids[i]][x_slice,y_slice,ti]
                zeros = (denom == 0) | (grid == 0)
                grid = grid/denom
                grid[zeros] = 0 # avoid nans
            image_at_time = np.flipud(grid.T)
            img_var = glm_image_to_var(image_at_time,
                                       field_name, description, units,
                                       ('y', 'x'))
            # Why does this line clear the attrs on the coords?
            # log.debug("*** Checking x coordinate attrs {0}a".format(i))
            # log.debug(dataset.x.attrs)
            dataset[img_var.name] = img_var
            # log.debug("*** Checking x coordinate attrs {0}b".format(i))
            # log.debug(dataset.x.attrs)

        # Restore the cleared coord attrs
        dataset.x.attrs.update(xattrs)
        dataset.y.attrs.update(yattrs)

        # log.debug("*** Checking x coordinate attrs final")
        # log.debug(dataset.x.attrs)
            
        log.info("Preparing to write NetCDF {0}".format(outfile))
        dataset.to_netcdf(outfile)
        log.info("Wrote NetCDF {0}".format(outfile))
