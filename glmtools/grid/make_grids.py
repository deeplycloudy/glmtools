""" Gridding of GLM data built on lmatools

"""
import numpy as np
from glmtools.io.mimic_lma import read_flashes
from glmtools.io.glm import GLMDataset
from glmtools.grid.clipping import QuadMeshSubset
from lmatools.grid.make_grids import FlashGridder
from lmatools.grid.fixed import get_GOESR_coordsys
import sys
    
class GLMGridder(FlashGridder):
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

