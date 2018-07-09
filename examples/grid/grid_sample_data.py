import os, tempfile
import numpy as np
from numpy.testing import assert_equal

from glmtools.test.common import get_sample_data_path

sample_path = get_sample_data_path()
samples = [
    "OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc",
]
samples = [os.path.join(sample_path, s) for s in samples]


from make_GLM_grids import create_parser, grid_setup

def test_grid_sample_data():
    """ This is equivalent to running the following bash script
    
    python make_GLM_grids.py -o /path/to/output/ 
    --fixed_grid --split_events \
    --goes_position east --goes_sector conus \
    --dx=2.0 --dy=2.0 \
    --ctr_lon=0.0 --ctr_lat=0.0 \
    # --start=2018-07-02T04:33:00 --end=2018-07-02T04:34:00 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \
    
    start and end are skipped, since they are determined from the filenames
    """
    resulting_date_path = ('2018', 'Jul', '02')
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
    with tempfile.TemporaryDirectory() as tmpdirname:
        cmd_args = ["-o", tmpdirname,
            "--fixed_grid", "--split_events",
            "--goes_position", "east", "--goes_sector", "conus", 
            "--dx=2.0", "--dy=2.0",
            "--ctr_lon=0.0", "--ctr_lat=0.0",] + samples
    
        parser = create_parser()
        args = parser.parse_args(cmd_args)
    
        from multiprocessing import freeze_support
        freeze_support()
        gridder, glm_filenames, start_time, end_time, grid_kwargs = grid_setup(args)
        gridder(glm_filenames, start_time, end_time, **grid_kwargs)
                
        for entry in os.scandir(os.path.join(tmpdirname, *resulting_date_path)):
            assert output_sizes[entry.name] == entry.stat().st_size
            # print(entry.name, entry.stat().st_size)