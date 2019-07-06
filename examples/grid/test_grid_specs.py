import os, tempfile
import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from glmtools.test.common import get_sample_data_path
from glmtools.grid.make_grids import subdivided_fixed_grid
from lmatools.grid.fixed import get_GOESR_grid

sample_path = get_sample_data_path()
samples = [
    "OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc",
]
samples = [os.path.join(sample_path, s) for s in samples]

from make_GLM_grids import create_parser, grid_setup


def grid_sample_data(grid_spec, dirname=None):
    """
    grid_spec are all command line arguments except for:
        -o dir_name
        input data filenames

    generates (yields) a sequence of filenames for each created file.
    """
    resulting_date_path = ('2018', 'Jul', '02')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdirname = os.path.join(tmpdir, dirname)
        cmd_args = ["-o", tmpdirname] + grid_spec + samples
    
        parser = create_parser()
        args = parser.parse_args(cmd_args)
    
        from multiprocessing import freeze_support
        freeze_support()
        gridder, glm_filenames, start_time, end_time, grid_kwargs = grid_setup(args)
        gridder(glm_filenames, start_time, end_time, **grid_kwargs)
        
        # gridder(glm_filenames, start_time, end_time, **grid_kwargs)
        for entry in os.scandir(os.path.join(tmpdirname, *resulting_date_path)):
            yield entry

def check_against_pug(check, pug_spec):
    """ check is an xarray dataset.
        pug_spec is as returned by get_GOESR_grid
    """
    x = check.x.data
    y = check.y.data
    
    nx_valid = pug_spec['pixelsEW']
    ny_valid = pug_spec['pixelsNS']
    nx_actual, ny_actual = check.x.shape[0], check.y.shape[0]
    assert(nx_actual==nx_valid)
    assert(ny_actual==ny_valid)

    # A simple delta of the first and second has some numerical noise in
    # it, but taking the mean eliminates it.
    dx_actual = np.abs((x[1:] - x[:-1]).mean())
    dy_actual = np.abs((y[1:] - y[:-1]).mean())
    np.testing.assert_allclose(dx_actual, pug_spec['resolution'], rtol=1e-5)
    np.testing.assert_allclose(dy_actual, pug_spec['resolution'], rtol=1e-5)

    # the x, y coordinates are the center points of the pixels, so
    # the span of the image (to the pixel corners) is an extra pixel
    # in each direction (1/2 pixel on each edge)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xspan_actual = np.abs(xmax - xmin) + dx_actual
    yspan_actual = np.abs(ymax - ymin) + dy_actual
    np.testing.assert_allclose(xspan_actual, pug_spec['spanEW'], rtol=1e-5)
    np.testing.assert_allclose(yspan_actual, pug_spec['spanNS'], rtol=1e-5)
    
    x_center_right = xmax -((xspan_actual-dx_actual)/2.0)
    y_center_right = ymax -((yspan_actual-dy_actual)/2.0)
    x_center_left = xmin + ((xspan_actual-dx_actual)/2.0)
    y_center_left = ymin + ((yspan_actual-dy_actual)/2.0)
    np.testing.assert_allclose(x_center_right, pug_spec['centerEW'], 
        rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(y_center_right, pug_spec['centerNS'], 
        rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(x_center_left, pug_spec['centerEW'], 
        rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(y_center_left, pug_spec['centerNS'], 
        rtol=1e-5, atol=1e-8)

def test_conus_size():
    """
    Test that calling examples/grid/make_glm_grids.py creates a CONUS domain
    that matches the sizes described in the GOES-R PUG.

    """
    position = 'east'
    view = 'conus'
    resolution = '2.0km'

    pug_spec = get_GOESR_grid(position=position, view=view,
                    resolution=resolution)
    grid_spec = ["--fixed_grid", "--split_events",
                "--goes_position", position, "--goes_sector", view,
                "--dx={0}".format(resolution[:-2]),
                "--dy={0}".format(resolution[:-2])]
    for entry in grid_sample_data(grid_spec, dirname=view):
        check = xr.open_dataset(entry.path)
        check_against_pug(check, pug_spec)

def test_meso_size():
    """
    Test that calling examples/grid/make_glm_grids.py creates a Meso domain
    that matches the sizes described in the GOES-R PUG.

    """
    position = 'east'
    view = 'meso'
    resolution = '2.0km'
    
    pug_spec = get_GOESR_grid(position=position, view=view,
                    resolution=resolution)
                    
    # For the GOES-E and a ctr_lon (below) of -75 (nadir) the center
    # will be 0,0
    pug_spec['centerEW'] = 0.0
    pug_spec['centerNS'] = 0.0
    
    grid_spec = ["--fixed_grid", "--split_events",
                "--goes_position", position, "--goes_sector", view, 
                "--dx={0}".format(resolution[:-2]), 
                "--dy={0}".format(resolution[:-2]),
                "--ctr_lat=0.0", "--ctr_lon=-75.0",
            ]
    for entry in grid_sample_data(grid_spec, dirname=view):
        check = xr.open_dataset(entry.path)
        check_against_pug(check, pug_spec)

        # For a mesoscale domain centered on the satellite subpoint the
        # east and west deltas should be symmetric, and the center should
        # be directly below the satellite.
        np.testing.assert_allclose(check.x[-1], -check.x[0], rtol=1e-5)
        np.testing.assert_allclose(check.y[-1], -check.y[0], rtol=1e-5)
