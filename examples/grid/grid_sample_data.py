import os, tempfile
import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from glmtools.io.glm import GLMDataset
from glmtools.test.common import get_sample_data_path

sample_path = get_sample_data_path()
samples = [
    "OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc",
]
samples = [os.path.join(sample_path, s) for s in samples]

def total_energy_in_L2(files, lon_range = (-102.5, -99.5),
                       lat_range = (31, 35)):
    # Gather up all flashes over West Texas, which is all flashes in the
    # mesoscale domain centered on that location at the time of the sample fiels
    energy = 0.0
    for sf in files:
        glm = GLMDataset(sf)
        flashes_subset = glm.subset_flashes(lon_range = lon_range, 
                                            lat_range = lat_range)
        energy += flashes_subset.event_energy.sum().data
    return energy


from make_GLM_grids import create_parser, grid_setup

def compare_to_sample_grids(tmpdirname, resulting_date_path, output_sizes):
    for entry in os.scandir(os.path.join(tmpdirname, *resulting_date_path)):
        target = output_sizes[entry.name]
        actual = entry.stat().st_size
        percent = 1
        assert  np.abs(target-actual) < int(target*percent/100)
        # print(entry.name, entry.stat().st_size)

def grid_sample_data(grid_spec, output_sizes, dirname=None):
    """
    grid_spec are all command line arguments except for:
        -o dir_name
        input data filenames

    output_sizes is a dictionary mapping from output filename to the expected
        size of the file, which may vary by up to 20%
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
        
        # print("Output file sizes")
        for entry in os.scandir(os.path.join(tmpdirname, *resulting_date_path)):
            
            # File size should be close to what we expect, with some platform
            # differences due to OS, compression, etc.
            target = output_sizes[entry.name]
            actual = entry.stat().st_size
            percent = 1
            assert  np.abs(target-actual) < int(target*percent/100)

            # Now compare the contents directly
            valid_file = os.path.join(sample_path, dirname, 
                                      *resulting_date_path, entry.name)
            valid = xr.open_dataset(valid_file)
            check = xr.open_dataset(entry.path)
            xr.testing.assert_allclose(valid, check)
            
            if (('total_energy' in output_sizes) & 
                ('total_energy' in entry.name)):
                np.testing.assert_allclose(check.total_energy.sum().data,
                                           output_sizes['total_energy'],
                                           rtol=1e-6)
                # An abundance of caution: validate the valid data, too!
                np.testing.assert_allclose(valid.total_energy.sum().data,
                                           output_sizes['total_energy'],
                                           rtol=1e-6)

            # for valid_var, valid_data in valid.variables.items():
            #     check_data = check.variables[valid_var]
            #     assert numpy.all_close(valid_data.data[:], check_data.data[:])

            # print(entry.name, entry.stat().st_size)

def test_fixed_grid_conus():
    """ This is equivalent to running the following bash script, which produces
    GLM grids with split events on the fixed grid at 2 km resolution in a
    CONUS sector.
    
    python make_GLM_grids.py -o /path/to/output/ 
    --fixed_grid --split_events \
    --goes_position east --goes_sector conus \
    --dx=2.0 --dy=2.0 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \
    
    start and end are sdetermined from the filenames
    ctr_lon and ctr_lat are implicit in the CONUS grid
    """
    
    grid_spec = ["--fixed_grid", "--split_events",
                "--goes_position", "east", "--goes_sector", "conus", 
                "--dx=2.0", "--dy=2.0",
                # "--ctr_lon=0.0", "--ctr_lat=0.0",
                ]
    output_sizes = {
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_extent.nc':123145,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_init.nc':59715,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_footprint.nc':102662,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_area.nc':113400,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_extent.nc':128665,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_init.nc':63054,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_source.nc':128538,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_total_energy.nc':133139,
    }

    grid_sample_data(grid_spec, output_sizes, dirname='conus')

def test_fixed_grid_arbitrary():
    """ This is equivalent to running the following bash script, which produces
    GLM grids with split events on the fixed grid at 2 km resolution in a
    sector defined by a center lon/lat point and a width and height.

    python make_GLM_grids.py -o /path/to/output/
    --fixed_grid --split_events \
    --goes_position east --goes_sector conus \
    --dx=2.0 --dy=2.0 "--width=1000.0", "--height=500.0" \
    --ctr_lon=0.0 --ctr_lat=0.0 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \

    start and end are skipped, since they are determined from the filenames
    """

    grid_spec = ["--fixed_grid", "--split_events",
                "--goes_position", "east",
                "--dx=2.0", "--dy=2.0",
                "--ctr_lat=33.5", "--ctr_lon=-101.5",
                "--width=1000.0", "--height=500.0"
                ]
    output_sizes = {
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_extent.nc':24775,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_init.nc':19576,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_footprint.nc':22235,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_area.nc':23941,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_extent.nc':25421,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_init.nc':19838,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_source.nc':25294,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_total_energy.nc':25820,
    }

    grid_sample_data(grid_spec, output_sizes, dirname='customsize')
    
def test_fixed_grid_meso():
    """ This is equivalent to running the following bash script, which produces
    GLM grids with split events on the fixed grid at 2 km resolution in a
    sector defined by a center lon/lat point and a width and height given by
    using the mesoscale lookup.

    python make_GLM_grids.py -o /path/to/output/
    --fixed_grid --split_events \
    --goes_position east --goes_sector meso \
    --dx=2.0 --dy=2.0 \
    --ctr_lon=-101.5 --ctr_lat=33.5 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \

    start and end are determined from the filenames.
    """

    grid_spec = ["--fixed_grid", "--split_events",
                "--goes_position", "east", "--goes_sector", "meso",
                "--dx=2.0", "--dy=2.0",
                "--ctr_lat=33.5", "--ctr_lon=-101.5",
                ]
    output_sizes = {
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_extent.nc':26370,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_flash_init.nc':21052,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_footprint.nc':23939,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_area.nc':25637,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_extent.nc':27053,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_group_init.nc':21305,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_source.nc':26926,
        'GLM-00-00_20180702_043300_60_1src_056urad-dx_total_energy.nc':27501,
        'total_energy':total_energy_in_L2(samples),
    }

    grid_sample_data(grid_spec, output_sizes, dirname='meso')