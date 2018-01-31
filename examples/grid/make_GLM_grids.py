
import argparse
parse_desc = """Grid GLM flash data. The start and end times can be specified
independently, or if not provided they will be inferred from the filenames.

Grid spacing is regular in latitude and longitude with the grid box
being sized to match the requested dx, dy at the center of the grid.

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
                    dest='ctr_lat', action='store', type=float,
                    help='center latitude')
parser.add_argument('--ctr_lon', metavar='longitude', required=True,
                    dest='ctr_lon', action='store', type=float,
                    help='center longitude')
parser.add_argument('--start', metavar='yyyy-mm-ddThh:mm:ss',
                    dest='start', action='store', 
                    help='UTC start time, e.g., 2017-07-04T08:00:00')
parser.add_argument('--end', metavar='yyyy-mm-ddThh:mm:ss',
                    dest='end', action='store', 
                    help='UTC end time, e.g., 2017-07-04T09:00:00')
parser.add_argument('--dx', metavar='km', 
                    dest='dx', action='store', default=10.0, type=float,
                    help='approximate east-west grid spacing')
parser.add_argument('--dy', metavar='km', 
                    dest='dy', action='store', default=10.0, type=float,
                    help='approximate north-south grid spacing')
parser.add_argument('--dt', metavar='seconds', 
                    dest='dt', action='store', default=60.0, type=float,
                    help='frame duration')
parser.add_argument('--width', metavar='distance in km', 
                    dest='width', action='store', default=400.0, type=float,
                    help='total width of the grid')
parser.add_argument('--height', metavar='distance in km', 
                    dest='height', action='store', default=400.0, type=float,
                    help='total height of the grid')
parser.add_argument('--nevents', metavar='minimum events per flash', type=int,
                    dest='min_events', action='store', default=1,
                    help='minimum number of events per flash')
parser.add_argument('--ngroups', metavar='minimum groups per flash', type=int,
                    dest='min_groups', action='store', default=1,
                    help='minimum number of groups per flash')
parser.add_argument('--fixed_grid',
                    action='store_true', dest='fixed_grid',
                    help='grid to the geostationary fixed grid')
parser.add_argument('--goes_position', default='none',
                    action='store', dest='goes_position',
                    help='one of [east|west|test]. also requires goes_sector.')
parser.add_argument('--goes_sector', default='none',
                    action='store', dest='goes_sector',
                    help='one of [full|conus|meso]. also requires goes_position. If sector is meso, ctr_lon and ctr_lat are interpreted as the ctr_x and ctr_y of the fixed grid')
parser.add_argument('--corner_points', metavar='filename.pickle',
                    action='store', dest='corner_points', 
                    help='name of file containing a pickled corner point lookup table')
parser.add_argument('--split_events', dest='split_events', 
                    action='store_true',
                    help='Split GLM event polygons when gridding')
parser.add_argument('--lma', dest='is_lma', 
                    action='store_true', 
                    help='grid LMA h5 files instead of GLM data')
# parser.add_argument('-v', dest='verbose', action='store_true',
                    # help='verbose mode')
# parser.add_argument('--speed', dest='speed', action='store',
#                     choices={'slow','fast'}, default='slow',
#                     help='search speed')
args = parser.parse_args()
    
##### END PARSING #####

import numpy as np
import subprocess, glob
from datetime import datetime
import os
from functools import partial

from lmatools.grid.make_grids import write_cf_netcdf_latlon, write_cf_netcdf_noproj, write_cf_netcdf_fixedgrid
from lmatools.grid.make_grids import dlonlat_at_grid_center, grid_h5flashfiles
from glmtools.grid.make_grids import grid_GLM_flashes
from glmtools.io.glm import parse_glm_filename
from lmatools.io.LMA_h5_file import parse_lma_h5_filename
from lmatools.grid.fixed import get_GOESR_grid, get_GOESR_coordsys

# When passed None for the minimum event or group counts, the gridder will skip 
# the check, saving a bit of time.
min_events = int(args.min_events)
if min_events <= 1:
    min_events = None
min_groups = int(args.min_groups)
if min_groups <= 1:
    min_groups = None

if args.is_lma:
    filename_parser = parse_lma_h5_filename
    start_idx = 0
    end_idx = 1
else:
    filename_parser = parse_glm_filename
    start_idx = 3
    end_idx = 4
    
glm_filenames = args.filenames
base_filenames = [os.path.basename(p) for p in glm_filenames]
try:
    filename_infos = [filename_parser(f) for f in base_filenames]
    # opsenv, algorithm, platform, start, end, created = parse_glm_filename(f)
    filename_starts = [info[start_idx] for info in filename_infos]
    filename_ends = [info[end_idx] for info in filename_infos]
except ValueError:
    print("One or more GLM files has a non-standard filename.")
    print("    Assuming that --start and --end have been passed directly.")
    
from glmtools.io.glm import parse_glm_filename
if args.start is not None:
    start_time = datetime.strptime(args.start[:19], '%Y-%m-%dT%H:%M:%S')
else:
    start_time = min(filename_starts)
if args.end is not None:
    end_time = datetime.strptime(args.end[:19], '%Y-%m-%dT%H:%M:%S')
else:
    end_time = max(filename_ends)

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
# lon lat tuples of the corners
corners = np.vstack([(x_bnd[0], y_bnd[0]), (x_bnd[0], y_bnd[1]), 
                     (x_bnd[1], y_bnd[1]), (x_bnd[1], y_bnd[0])])
# print(x_bnd, y_bnd)

# Default
proj_name='latlong'
output_writer = write_cf_netcdf_latlon

# Old support for a direct lon/lat to fixed grid lookup table. Unused
# at the moment.
pixel_grid = None
if pixel_grid is not None:
    from glmtools.io.ccd import load_pixel_lookup
    from lmatools.coordinateSystems import PixelGrid
    ccd_lookup_data = load_pixel_lookup(args.fixed_grid)
    ccd_lookup, ccdX, ccdY, ccd_lons, ccd_lats = ccd_lookup_data
    ccd_dist, ccd_idx = ccd_lookup.query(corners)
    corner_ccdX, corner_ccdY = ccdX[ccd_idx], ccdY[ccd_idx]
    ccd_Xmin, ccd_Ymin = corner_ccdX.min(), corner_ccdY.min()
    ccd_Xmax, ccd_Ymax = corner_ccdX.max(), corner_ccdY.max()
    ccd_grid = PixelGrid(ccd_lons, ccd_lats, ccd_lookup, ccdX, ccdY)
    # if we want indices 1, 2, 3 for min and mix of 1, 3 with dx=1
    # then in gridder xedge must be .5, 1.5, 2.5, 3.5 to give centers 1,2,3.
    # Gridder starts at x_bnd[0], so subtract .5
    # and then goes to x_bnd[1]+dx by dx, so add 1 to get 4
    # giving np.arange(.5, 4, 1.0) == array([ 0.5,  1.5,  2.5,  3.5])
    x_bnd = ccd_Xmin-.5, ccd_Xmax+1
    y_bnd = ccd_Ymin-.5, ccd_Ymax+1
    dx, dy = 1, 1
    output_writer = write_cf_netcdf_noproj
    proj_name='pixel_grid'
    # ccdsub_lons=lons[ccd_Xmin:ccd_Xmax+1, ccd_Ymin:ccd_Ymax+1]
    # ccdsub_lats=lats[ccd_Xmin:ccd_Xmax+1, ccd_Ymin:ccd_Ymax+1]
# print(x_bnd, y_bnd)

if args.fixed_grid:
    proj_name = 'geos'
    if (args.goes_position != 'none') & (args.goes_sector != 'none'):
        goes_resln_options = np.asarray([0.5, 1.0, 2.0, 4.0, 10.0])
        resln_idx = np.argmin(np.abs(goes_resln_options - args.dx))
        closest_resln = goes_resln_options[resln_idx]
        resln = '{0:4.1f}km'.format(closest_resln).replace(' ', '')
        view = get_GOESR_grid(position=args.goes_position, 
                              view=args.goes_sector, 
                              resolution=resln)
        nadir_lon = view['nadir_lon']
        dx = dy = view['resolution']
        nx, ny = view['pixelsEW'], view['pixelsNS']
        if 'centerEW' in view:
            x_ctr, y_ctr = view['centerEW'], view['centerNS']
        else:
            # won't be known for mesoscale sectors. 
            # Assume that it's in ctr_lon, ctr_lat
            x_ctr, y_ctr = ctr_lon, ctr_lat
        x_bnd = (np.arange(nx, dtype='float') - nx/2.0)*dx + x_ctr + 0.5*dx
        y_bnd = (np.arange(ny, dtype='float') - ny/2.0)*dy + y_ctr + 0.5*dy
        x_bnd = np.asarray([x_bnd.min(), x_bnd.max()])
        y_bnd = np.asarray([y_bnd.min(), y_bnd.max()])
        
        geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
        ctr_lon, ctr_lat, ctr_alt = grs80lla.fromECEF(
            *geofixcs.toECEF(x_ctr, y_ctr, 0.0))
        fixed_grid = geofixcs
        print(x_bnd, y_bnd, dx, dy, nx, ny)
        
        # Does this work? Almost - the fixed grid is actually a special
        # projection.
        output_writer = partial(write_cf_netcdf_fixedgrid, nadir_lon=nadir_lon)
        
        
if args.is_lma:
    gridder = grid_h5flashfiles
    output_filename_prefix='LMA'
else:
    gridder = grid_GLM_flashes
    output_filename_prefix='GLM'

grid_kwargs=dict(proj_name=proj_name,
        base_date = date, do_3d=False,
        dx=dx, dy=dy, frame_interval=frame_interval, x_bnd=x_bnd, y_bnd=y_bnd, 
        ctr_lat=ctr_lat, ctr_lon=ctr_lon, outpath = outpath,
        min_points_per_flash = min_events,
        output_writer = output_writer,
        output_filename_prefix=output_filename_prefix, spatial_scale_factor=1.0)

if args.fixed_grid:
    grid_kwargs['fixed_grid'] = True
    grid_kwargs['nadir_lon'] = nadir_lon
if args.split_events:
    grid_kwargs['clip_events'] = True
if min_groups is not None:
    grid_kwargs['min_groups_per_flash'] = min_groups
if args.is_lma:
    grid_kwargs['energy_grids'] = True
else:
    grid_kwargs['energy_grids'] = ('total_energy',)
if (proj_name=='pixel_grid') or (proj_name=='geos'):
    grid_kwargs['pixel_coords'] = fixed_grid
gridder(glm_filenames, start_time, end_time, **grid_kwargs)