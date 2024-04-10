import argparse
parse_desc = """Create GLM data quality product fields: flash DE and error flag image
as well as a scaled background image. Both at the 2 km fixed grid resolution.

Both images are 4-bit images, and so can be packed into a single byte.
"""

input_help = """name of a file with containing the list of data files
to be processed (possibly including full path), one data file per line"""

output_help = """Specify the output path and filename using a configurable path
template. -o ./{dataset_name} (the default) will generate files in the current
directory using a standard GOES imagery naming convention, including a .nc
extension."""

def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    # parser.add_argument(dest='filenames',metavar='filename', nargs='*')
    parser.add_argument('-f', '--file_of_data_files',
                        metavar='file with a list of input data files',
                        required=True, dest='file_of_files', action='store',
                        help=input_help)
    parser.add_argument('-o', '--output_path',
                        metavar='filename template including path',
                        required=False, dest='outdir', action='store',
                        default='./{dataset_name}', help=output_help)
    parser.add_argument('--packtobyte', dest='pack_byte',
                        action='store_true',
                        help='pack background and data quality image in one byte')
    
    return parser

# ===== end of parser config =====

from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from dqf import dqf_from_nav_background


def process_one_ds(ds_in, combine=False, outpath='./{dataset_name}'):
    # # All this will get replaced with output from Bitzer's code. 
    # # Note shapes of arrays should match below.
    import joblib
    lat_joblib, lon_joblib, back_in = joblib.load('/Users/ebruning/Downloads/nav_bkgnd.joblib')
    # Handle subsetting
    # back = back_in[:1299, :1370]
    lat_joblib.shape = (1299, 1370)
    lon_joblib.shape = (1299, 1370)
    
    
    # For now pretend the background image above is from the joblib file, since those are the only good positions we have
    # back_in.shape == (1300, 1372) from joblib, transposed from what we have here.
    # So need to transpose lat and lon, and also subset back_in to match those shapes
    lat = lat_joblib.T
    lon = lon_joblib.T
    back = ds_in.bg_dn.values[:1370, :1299]
    thresh_fJ = 1.0e15*ds_in.bg_img.values[:1370, :1299]
    # === end replacement ===
    # === Expected final code below ===
    # back = ds_in.bg_dn.values 
    # lat = ds_in.lats.values
    # lon = ds_in.lons.values
    # thresh_fJ = 1.0e15*ds_in.bg_img.values
    # === end expected final code ===
    
    start = datetime.strptime(ds_in.attrs['Date Valid'], '%Y-%m-%dT%H%M%S')
    end = start + timedelta(minutes=2, seconds=30)
    cache_key = ds_in.attrs['Navigation Parameter Retrieval']
    
    # Update dqf_from_nav_background to use other min and mix for back_cal,
    # which is as easy as changing the kwargs passed to scale_shift_back.
    # This must be changed if we start using an actually calibrated background,
    # instead of raw_dn (back, back as arguments below).
    outfile = dqf_from_nav_background(start, end, lat, lon, back, back, thresh_fJ, 
        cache_key=cache_key, combine_products=combine, outpath=outpath)
    
    
def main(infile, outpath, combine=False):
    with open(infile, 'r') as allfiles:
        for thisfile in allfiles:
            ds_in = xr.open_dataset(thisfile.strip()).load()
            process_one_ds(ds_in, combine=combine, outpath=outpath)
            

if __name__ == '__main__':
    import sys
    parser = create_parser()
    args = parser.parse_args()

    infile = args.file_of_files
    outpath = args.outdir
    main(infile, outpath, combine=args.pack_byte)
    
    
    