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
def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(dest='filenames',metavar='filename', nargs='*')
    parser.add_argument('-o', '--output_dir', metavar='directory',
                        required=True, dest='outdir', action='store', )
    parser.add_argument('--ctr_lat', metavar='latitude', required=False,
                        dest='ctr_lat', action='store', type=float,
                        help='center latitude')
    parser.add_argument('--ctr_lon', metavar='longitude', required=False,
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
                        dest='width', action='store', default=400.0,
                        type=float, help='total width of the grid')
    parser.add_argument('--height', metavar='distance in km', 
                        dest='height', action='store', default=400.0,
                        type=float, help='total height of the grid')
    parser.add_argument('--nevents', metavar='minimum events per flash',
                        type=int, dest='min_events', action='store', default=1,
                        help='minimum number of events per flash')
    parser.add_argument('--ngroups', metavar='minimum groups per flash',
                        type=int, dest='min_groups', action='store', default=1,
                        help='minimum number of groups per flash')
    parser.add_argument('--fixed_grid',
                        action='store_true', dest='fixed_grid',
                        help='grid to the geostationary fixed grid')
    parser.add_argument('--subdivide_grid', metavar='sqrt(number of subgrids)',
                        action='store', dest='subdivide_grid',
                        type=int, default=1,
                        help=("subdivide the grid this many times along "
                              "each dimension"))
    parser.add_argument('--goes_position', default='none',
                        action='store', dest='goes_position',
                        help=("One of [east|west|test]. "
                              "Also requires goes_sector."))
    parser.add_argument('--goes_sector', default='none',
                        action='store', dest='goes_sector',
                        help=("One of [full|conus|meso]. "
                              "Also requires goes_position. If sector is "
                              "meso, ctr_lon and ctr_lat are interpreted as "
                              "the ctr_x and ctr_y of the fixed grid"))
    parser.add_argument('--corner_points', metavar='filename.pickle',
                        action='store', dest='corner_points', 
                        help=("name of file containing a pickled "
                              "corner point lookup table"))
    parser.add_argument('--split_events', dest='split_events', 
                        action='store_true',
                        help='Split GLM event polygons when gridding')
    parser.add_argument('--ellipse', dest='ellipse_rev', default=-1,
                        action='store', type=int,
                        help='Lightning ellipse revision. -1 (default)=infer'
                             ' from date in each GLM file, 0=value at launch,'
                             ' 1=late 2018 revision')
    parser.add_argument('--lma', dest='is_lma', 
                        action='store_true', 
                        help='grid LMA h5 files instead of GLM data')
    # parser.add_argument('-v', dest='verbose', action='store_true',
                        # help='verbose mode')
    return parser
    
##### END PARSING #####

import numpy as np
import subprocess, glob
from datetime import datetime
import os
from functools import partial

import logging
class MyFormatter(logging.Formatter):
    """ Custom class to allow logging of microseconds"""
    converter=datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s
logoutfile = logging.FileHandler("make_GLM_grid.log")
formatter = MyFormatter(fmt='%(levelname)s %(asctime)s %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S.%f')
logoutfile.setFormatter(formatter)
logging.basicConfig(handlers = [logoutfile],
                    level=logging.DEBUG)

# Separate from log setup - actually log soemthign specific to this module.
log = logging.getLogger(__name__)
log.info("Starting GLM Gridding")

def nearest_resolution(args):
    """ Uses args.dx to find the closest resolution specified by the
    GOES-R PUG. Returns something like "10.0km" that can be used as the
    resolution argument to get_GOESR_grid.
    """
    goes_resln_options = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    resln_idx = np.argmin(np.abs(goes_resln_options - args.dx))
    closest_resln = goes_resln_options[resln_idx]
    resln = '{0:4.1f}km'.format(closest_resln).replace(' ', '')
    return resln

def grid_setup(args):
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
        log.error("One or more GLM files has a non-standard filename.")
        log.error("Assuming that --start and --end have been passed directly.")
    
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

    if args.fixed_grid:
        proj_name = 'geos'

        if (args.goes_position != 'none') & (args.goes_sector != 'none'):
            resln = nearest_resolution(args)
            view = get_GOESR_grid(position=args.goes_position, 
                                  view=args.goes_sector, 
                                  resolution=resln)
            nadir_lon = view['nadir_lon']
            dx = dy = view['resolution']
            nx, ny = view['pixelsEW'], view['pixelsNS']
            geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
            
            if 'centerEW' in view:
                x_ctr, y_ctr = view['centerEW'], view['centerNS']
            elif args.goes_sector == 'meso':
                # use ctr_lon, ctr_lat to get the center of the mesoscale FOV
                x_ctr, y_ctr, z_ctr = geofixcs.fromECEF(
                    *grs80lla.toECEF(args.ctr_lon, args.ctr_lat, 0.0))
        elif (args.goes_position != 'none') & (args.goes_sector == 'none'):
            # Requires goes_position, a center, and a width. Fully flexible
            # in resolution, i.e., doesn't slave it to one of the GOES-R specs
            view = get_GOESR_grid(position=args.goes_position, 
                                  view='full', 
                                  resolution='1.0km')
            nadir_lon = view['nadir_lon']
            dx1km = dy1km = view['resolution']
            geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
            x_ctr, y_ctr, z_ctr = geofixcs.fromECEF(
              *grs80lla.toECEF(args.ctr_lon, args.ctr_lat, 0.0))

            # Convert the specified resolution in km given by args.dx to
            # a delta in fixed grid coordinates using the 1 km delta from the
            # GOES-R PUG.
            dx, dy = args.dx * dx1km, args.dy * dy1km
            nx, ny = int(args.width/args.dx), int(args.height/args.dy)
        else:
            raise ValueError("Gridding on the fixed grid requires "
                "goes_position and dx. For goes_sector='meso', also specify "
                "ctr_lon and ctr_lat. Without goes_sector, also include width "
                "and height.")
        # Need to use +1 here to convert to xedge, yedge expected by gridder
        # instead of the pixel centroids that will result in the final image
        nx += 1
        ny += 1
        x_bnd = (np.arange(nx, dtype='float') - (nx)/2.0)*dx + x_ctr + 0.5*dx
        y_bnd = (np.arange(ny, dtype='float') - (ny)/2.0)*dy + y_ctr + 0.5*dy
        log.debug(("initial x,y_ctr", x_ctr, y_ctr))
        log.debug(("initial x,y_bnd", x_bnd.shape, y_bnd.shape))
        x_bnd = np.asarray([x_bnd.min(), x_bnd.max()])
        y_bnd = np.asarray([y_bnd.min(), y_bnd.max()])
        
        geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
        ctr_lon, ctr_lat, ctr_alt = grs80lla.fromECEF(
            *geofixcs.toECEF(x_ctr, y_ctr, 0.0))
        fixed_grid = geofixcs
        log.debug((x_bnd, y_bnd, dx, dy, nx, ny))

        output_writer = partial(write_cf_netcdf_fixedgrid, nadir_lon=nadir_lon)
    else:
        # Default
        proj_name='latlong'
        output_writer = write_cf_netcdf_latlon
        ctr_lat = float(args.ctr_lat)
        ctr_lon = float(args.ctr_lon)
        dx_km=float(args.dx)*1.0e3
        dy_km=float(args.dy)*1.0e3
        width, height = 1000.0*float(args.width), 1000.0*float(args.height)
        x_bnd_km = (-width/2.0, width/2.0)
        y_bnd_km = (-height/2.0, height/2.0)
        dx, dy, x_bnd, y_bnd = dlonlat_at_grid_center(ctr_lat, ctr_lon, 
                                    dx=dx_km, dy=dy_km,
                                    x_bnd = x_bnd_km, y_bnd = y_bnd_km )

    # tuples of the corners
    corners = np.vstack([(x_bnd[0], y_bnd[0]), (x_bnd[0], y_bnd[1]), 
                         (x_bnd[1], y_bnd[1]), (x_bnd[1], y_bnd[0])])
    # print(x_bnd, y_bnd)

    if args.is_lma:
        gridder = grid_h5flashfiles
        output_filename_prefix='LMA'
    else:
        gridder = grid_GLM_flashes
        output_filename_prefix='GLM'

    grid_kwargs=dict(proj_name=proj_name,
            base_date = date, do_3d=False,
            dx=dx, dy=dy, frame_interval=float(args.dt),
            x_bnd=x_bnd, y_bnd=y_bnd, 
            ctr_lat=ctr_lat, ctr_lon=ctr_lon, outpath = outpath,
            min_points_per_flash = min_events,
            output_writer = output_writer, subdivide=args.subdivide_grid,
            output_filename_prefix=output_filename_prefix,
            spatial_scale_factor=1.0)

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
    grid_kwargs['ellipse_rev'] = args.ellipse_rev
    # if args.corner_points:
        # grid_kwargs['corner_pickle'] = args.corner_points
    return gridder, glm_filenames, start_time, end_time, grid_kwargs

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    from multiprocessing import freeze_support
    freeze_support()
    gridder, glm_filenames, start_time, end_time, grid_kwargs = grid_setup(args)
    gridder(glm_filenames, start_time, end_time, **grid_kwargs)