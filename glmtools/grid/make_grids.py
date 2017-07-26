""" Gridding of GLM data built on lmatools

"""
from glmtools.io.mimic_lma import mimic_lma_dataset
from glmtools.io.glm import GLMDataset
from lmatools.grid.make_grids import FlashGridder
import sys

def read_flashes(glm, target, base_date=None, lon_range=None, lat_range=None,
                 min_events=None, min_groups=None):
    """ This routine is the data pipeline source, responsible for pushing out 
        events and flashes. Using a different flash data source format is a matter of
        replacing this routine to read in the necessary event and flash data."""
    
    if ((lon_range is not None) | (lat_range is not None) |
        (min_events is not None) | (min_groups is not None)):
        # only subset if we have to
        flash_data = glm.subset_flashes(lon_range=lon_range, lat_range=lat_range,
                        min_events=min_events, min_groups=min_groups)
    else:
        flash_data = glm.dataset

    try:
        events, flashes = mimic_lma_dataset(flash_data, base_date)
        if events.shape[0] >= 1:
            target.send((events, flashes))
            del events, flashes
    except KeyError as ke:
        err_txt = 'Skipping {0}\n    ... assuming a flash, group, or event with id {1} does not exist'
        print(err_txt.format(glm.dataset.dataset_name, ke))

    
class GLMGridder(FlashGridder):
    def process_flashes(self, glm, x_bnd=None, y_bnd=None, 
                        min_points_per_flash=1, min_groups_per_flash=1):
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
                     lon_range=x_bnd, lat_range=y_bnd)

        
def grid_GLM_flashes(GLM_filenames, start_time, end_time, **kwargs):
    """ Grid GLM data that has been converted to an LMA-like array format.
        
        Assumes that GLM data are being gridded on a lat, lon grid.
        
        Keyword arguments to this function
        are those to the FlashGridder class and its functions.
    """
    
    kwargs['do_3d'] = False
    
    process_flash_kwargs = {}
    for prock in ('min_points_per_flash','min_groups_per_flash'):
        # interpret x_bnd and y_bnd as lon, lat
        if prock in kwargs:
            process_flash_kwargs[prock] = kwargs.pop(prock)
    # need to also pass these kwargs through to the gridder for grid config.
    process_flash_kwargs['x_bnd'] = kwargs['x_bnd']
    process_flash_kwargs['y_bnd'] = kwargs['y_bnd']
            
    out_kwargs = {}
    for outk in ('outpath', 'output_writer', 'output_writer_3d', 'output_kwargs',
                 'output_filename_prefix', 'spatial_scale_factor'):
        if outk in kwargs:
            out_kwargs[outk] = kwargs.pop(outk)
    
    gridder = GLMGridder(start_time, end_time, **kwargs)
    for filename in GLM_filenames:
        print("Processing {0}".format(filename))
        sys.stdout.flush()
        glm = GLMDataset(filename)
        gridder.process_flashes(glm, **process_flash_kwargs)
        
    output = gridder.write_grids(**out_kwargs)
    return output    

