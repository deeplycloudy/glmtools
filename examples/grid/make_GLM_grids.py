
import argparse
parse_desc = """Grid GLM flash data

Grid spacing is regular in latitude and longitude with the grid box
being correctly sized at the center of the grid.

Within the output directory, a year/month/day directory will be created,
e.g., 2017/Jul/04/, and within that directory the grid files will be created.

Therefore, this script can be used to process multiple days and they will
be written to a standardized directory structure.
"""
parser = argparse.ArgumentParser(description=parse_desc)
parser.add_argument(dest='filenames',metavar='filename', nargs='*')
parser.add_argument('-o', '--output_dir', metavar='directory', required=True,
                    dest='outdir', action='store', )
parser.add_argument('--ctr_lat', metavar='latitude', required=True,
                    dest='ctr_lat', action='store',
                    help='center latitude')
parser.add_argument('--ctr_lon', metavar='longitude', required=True,
                    dest='ctr_lon', action='store',
                    help='center longitude')
parser.add_argument('--start', metavar='yyyy-mm-ddThh:mm:ss', required=True,
                    dest='start', action='store', 
                    help='UTC start time, e.g., 2017-07-04T08:00:00')
parser.add_argument('--end', metavar='yyyy-mm-ddThh:mm:ss', required=True,
                    dest='end', action='store', 
                    help='UTC end time, e.g., 2017-07-04T09:00:00')
parser.add_argument('--dx', metavar='km', 
                    dest='dx', action='store', default=10.0,
                    help='approximate east-west grid spacing')
parser.add_argument('--dy', metavar='km', 
                    dest='dy', action='store', default=10.0,
                    help='approximate north-south grid spacing')
parser.add_argument('--dt', metavar='seconds', 
                    dest='dt', action='store', default=60.0,
                    help='frame duration')
parser.add_argument('--width', metavar='distance in km', 
                    dest='width', action='store', default=400.0,
                    help='total width of the grid')
parser.add_argument('--height', metavar='distance in km', 
                    dest='height', action='store', default=400.0,
                    help='total height of the grid')
# parser.add_argument('-v', dest='verbose', action='store_true',
                    # help='verbose mode')
# parser.add_argument('--speed', dest='speed', action='store',
#                     choices={'slow','fast'}, default='slow',
#                     help='search speed')
args = parser.parse_args()
    
##### END PARSING #####

import subprocess, glob
from datetime import datetime
import os

from lmatools.grid.make_grids import write_cf_netcdf_latlon, dlonlat_at_grid_center, grid_h5flashfiles
from glmtools.grid.make_grids import grid_GLM_flashes

start_time = datetime.strptime(args.start[:19], '%Y-%m-%dT%H:%M:%S')
end_time = datetime.strptime(args.end[:19], '%Y-%m-%dT%H:%M:%S')
glm_filenames = args.filenames

date = datetime(start_time.year, start_time.month, start_time.day)
# grid_dir = os.path.join('/data/LCFA-production/', 'grid_test')
# outpath = grid_dir+'/20%s' %(date.strftime('%y/%b/%d'))
outpath = os.path.join(args.outdir, '20%s' %(date.strftime('%y/%b/%d')))
if os.path.exists(outpath) == False:
    os.makedirs(outpath)
    # subprocess.call(['chmod', 'a+w', outpath, grid_dir+'/20%s' %(date.strftime('%y/%b')), grid_dir+'/20%s' %(date.strftime('%y'))])

# center_ID='WTLMA'
ctr_lat = float(args.ctr_lat)
ctr_lon = float(args.ctr_lon)
dx_km=float(args.dx)*1.0e3
dy_km=float(args.dy)*1.0e3
width, height = 1000.0*float(args.width), 1000.0*float(args.height)
x_bnd_km = (-width/2.0, width/2.0)
y_bnd_km = (-height/2.0, height/2.0)
frame_interval = float(args.dt)

dx, dy, x_bnd, y_bnd = dlonlat_at_grid_center(ctr_lat, ctr_lon, 
                            dx=dx_km, dy=dy_km,
                            x_bnd = x_bnd_km, y_bnd = y_bnd_km )
print(x_bnd, y_bnd)

grid_GLM_flashes(glm_filenames, start_time, end_time, proj_name='latlong',
        base_date = date, do_3d=False,
        dx=dx, dy=dy, frame_interval=frame_interval,
        #dz=dz, z_bnd=z_bnd_km,
        x_bnd=x_bnd, y_bnd=y_bnd, 
        ctr_lat=ctr_lat, ctr_lon=ctr_lon, outpath = outpath,
        output_writer = write_cf_netcdf_latlon,
        output_filename_prefix='GLM', spatial_scale_factor=1.0
        )