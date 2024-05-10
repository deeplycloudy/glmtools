import os

import numpy as np
import scipy.interpolate as spint
# from scipy.spatial import Delaunay

import matplotlib.colors as colors

from lmatools.grid.fixed import get_GOESR_grid, get_GOESR_coordsys
from glmtools.io.lightning_ellipse import ltg_ellps_lon_lat_to_fixed_grid
from glmtools.io.imagery import glm_image_to_var, get_glm_global_attrs, new_goes_imagery_dataset



back_mask = int('11110000', 2)
dqp_mask = int('00001111', 2)

# FDE will be the eight values in the lowest three bits
# Levels are the edges of the color bins
fde_levels = [0, 20, 40, 50, 60, 70, 80, 90, 100]
fde_colors = [
    (0.25,0.12,0.66), (0.29,0.22,0.89), 
    (0.29,0.36,1.0), (0.2,0.5,1.0), 
    (0.16,0.62,0.9), (0.05,0.72,0.8),
    (0.19,0.77,0.64), (0.4,0.81,0.43),
]

fde_labels = ['(' + str(s[1]) +') FDE >' + str(s[0]) +'%' for s in zip(fde_levels[:-1], range(8))]

# Use matplotlib's colormapping capability to make it easy to
# assign a value to a non-uniform range
fde_cmap, fde_norm = colors.from_levels_and_colors(
    fde_levels, fde_colors, extend = 'neither')
    
    
dead_color = (0.1, 0.1, 0.1) # dark gray - black
obscured_color = (0.5, 0.5, 0.5) # gray
hardware_dropped_color = (0.65, 0.16, 0.09) # dark red
algorithm_dropped_color = (0.95, 0.65, 0.57) # pink

intrusion_color = (0.89, 0.47, 0.18) # dark orange
glint_color = (0.95, 0.70, 0.24) # orange
near_saturation_color = (0.99, 0.95, 0.44) # yellow
at_saturation_color = (1.0, 1.0, 1.0) # white

# All flags will have the fourth bit set, 
# and then use the 8 unique values in the lower three bits
flag_shift_bits = 3
flag_shift_amount = 2**flag_shift_bits -1
dead_val = 8 + flag_shift_amount
obscured_val = 7 + flag_shift_amount
hardware_dropped_val = 6 + flag_shift_amount
algorithm_dropped_val = 5 + flag_shift_amount
intrusion_val = 4 + flag_shift_amount
glint_val = 3 + flag_shift_amount
near_saturation_val = 2 + flag_shift_amount
at_saturation_val = 1 + flag_shift_amount

flag_colors = [dead_color, obscured_color,
               hardware_dropped_color, algorithm_dropped_color,
               intrusion_color, glint_color,
               near_saturation_color, at_saturation_color][::-1]

flag_labels = ['Dead pixel', 'Obscured pixel', 
    'Dropped event - hardware', 'Dropped event - software',
    'Solar intrusion', 'Solar glint' , 
    'Near saturation', 'At saturation'][::-1]

flag_values = [dead_val, obscured_val,
               hardware_dropped_val, algorithm_dropped_val,
               intrusion_val, glint_val,
               near_saturation_val, at_saturation_val][::-1]
               
dqp_labels = fde_labels + ['(' + str(s[1]) +') ' + s[0] for s in zip(flag_labels, flag_values)]

back_colors = [(1.0 - .06*i,)*3 for i in range(16)][::-1]


def scale_fde(fde):
    scaled_fde = fde_norm(fde)
    # Ensure no values exceed the max or min in the colormap
    # typically, less than 0 to 0, and greater than 100 to 100
    fde_large = fde >= max(fde_levels)
    fde_small = fde < min(fde_levels)
    scaled_fde[fde_large] = fde_cmap.N - 1
    scaled_fde[fde_small] = 0
    return scaled_fde
      

# The remaining four bits are the backgrounds
def scale_shift_back(back, back_min = 0, back_max = 13000, shift=True):

    # Ensure no values exceed the max or min
    back_large = back >= back_max
    back_small = back < back_min
    
    field_bits = 8
    back_bits = 4
    back_scaled_max = (2**back_bits)-1

    back_normed = (float(back_scaled_max)*(back - back_min).astype(float)/float(back_max-back_min)).astype('u1')

    back_normed[back_large] = back_scaled_max
    back_normed[back_small] = 0

    if shift == True:
        back_bit_offset = field_bits - back_bits
        return np.left_shift(back_normed, back_bit_offset)
    else:
        return back_normed

# thresh_to_de_data = pd.read_csv('threshold_to_DE.csv')

def thresh_to_de(thresh):
    """ Convert thresh in fJ to flash detection efficiency in %.
    
    Typicaly input values are of order 1 fJ.
    
    This function is a polynomial fit to the empirical curve determined by Ken Cummins
    (2020, personal comm., GLM Sci. Mtg.), from a comparision of long-term statistics of
    LIS group energy.
    """
    coeffs_excel_cursed = [3.0e-5, -0.0028, 0.0930, -1.2706, 1.5, 100]
    coeffs_excel_less_cursed = [0.0000283755, -0.0026334487, 0.0893214041, -1.2187248667, 1.1917128155, 100.5631379079]

    # These are very close to the re-fit excel numbers. Use these.
    coeffs_numpy = [9.08015710e-04, -3.98651393e-02,  6.32570915e-01, -3.86511037e+00,
       -1.97632925e+00,  1.00622842e+02]
    
    coeffs = coeffs_numpy
    
    thresh_out_of_range = (thresh > 14.5)
    
    de = (  coeffs[0] * thresh*thresh*thresh*thresh*thresh
          + coeffs[1] * thresh*thresh*thresh*thresh
          + coeffs[2] * thresh*thresh*thresh
          + coeffs[3] * thresh*thresh
          + coeffs[4] * thresh 
          + coeffs[5])
    # if isinstance(thresh, xr.DataArray):
        # de.where(thresh_out_of_range, other=7.0)
    # else:
    de[thresh_out_of_range] = 7.0
    de[de>100] = 100.0
    return de

def get_fixed_grid_coords():
    view = get_GOESR_grid(position='east', view='full', resolution='2.0km')

    nadir_lon = view['nadir_lon']
    dx = dy = view['resolution']
    nx, ny = view['pixelsEW'], view['pixelsNS']
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
    x_ctr, y_ctr = view['centerEW'], view['centerNS']

    # Need to use +1 here to convert to xedge, yedge expected by gridder
    # instead of the pixel centroids that will result in the final image
    nx += 1
    ny += 1
    x_bnd = (np.arange(nx, dtype='float') - (nx)/2.0)*dx + x_ctr + 0.5*dx
    y_bnd = (np.arange(ny, dtype='float') - (ny)/2.0)*dy + y_ctr + 0.5*dy
    # Now get the centers as the middle of the edges
    x_ctr = (x_bnd[1:] + x_bnd[:-1])/2.0
    y_ctr = (y_bnd[1:] + y_bnd[:-1])/2.0

    Y, X = np.meshgrid(y_ctr, x_ctr)

    # geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
    # ctr_lon, ctr_lat, ctr_alt = grs80lla.fromECEF(
    #     *geofixcs.toECEF(x_ctr, y_ctr, 0.0))

    # print(x_bnd.shape, y_bnd.shape)
    # print(x_ctr.shape, y_ctr.shape)
    # print(X.shape, Y.shape)
    
    return x_bnd, y_bnd, x_ctr, y_ctr, X, Y
    
def interpolate_ccd_to_fixed_grid(data, x, y, X, Y, cache_key, cache_path='./', refresh_cache=False):
    """
    x, y are 2D arrays of CCD x and y positions
    X, Y are 2D arrays of interpolation positions
    
    
    cache_key is any string (e.g., an hour of the day such as '03'). 
    It will be used to create a file at cache_path (e.g., './03.npy')
    that saves the nearest neighbor interpolation, greatly speeding up successive runs.
    
    Setting refresh_cache 
    """
    
    subset = slice(None,None,1)
    # (N, D)
    interp_loc = np.vstack((X[subset, subset].flatten(), Y[subset, subset].flatten())).T
    data_loc = np.vstack((x[subset,subset].flatten(), y[subset,subset].flatten())).T
    
    # Check for finite (not-nan) values of data locations
    data_keep = ((np.isfinite(data_loc[:,0])) & (np.isfinite(data_loc[:,1])))
    data_loc = data_loc[data_keep, :] # second dimension has N=2, i.e., [n_points, 2]

    interp_data = data[subset,subset].flatten()[data_keep]

    
    # Thought about caching the cKDTree used in
    # https://github.com/scipy/scipy/blob/v1.12.0/scipy/interpolate/_ndgriddata.py#L20-L166
    # But setting up the kdtree is much faster than the querying part.
    # import pickle
    # import scipy.spatial
    # tree=scipy.spatial.cKDTree(data_loc)
    # pickle.dump(tree,open('tree.p','wb'))

    # Instead of interpolating the values, interpolate the index of the data value.
    # This will allow us to use the indexes as a cache of how to do the nearest neighbor lookup
    
    cache_file = os.path.join(cache_path, cache_key+'.npy')  
    if os.path.exists(cache_file) & (refresh_cache == False):
        # Have a cached file
        indexed_interp_field = np.load(cache_file)            
    else:
        # Create and cache the interpolation lookup from scratch.
        # Set up tree in a way that is a faster in query (2x in my experimentation), per
        # https://stackoverflow.com/questions/31819778/scipy-spatial-ckdtree-running-slowly/31840982#31840982
        tree_options = dict(balanced_tree=False, compact_nodes=False)
        interp_data_index = np.arange(interp_data.shape[0])
        index_interpolator = spint.NearestNDInterpolator(data_loc, interp_data_index, tree_options=tree_options)
        indexed_interp_field = index_interpolator(interp_loc)
        np.save(cache_file, indexed_interp_field)
    
    # Now use the index cache to look up the original data values
    # out_field = np.empty(interp_loc.shape[0])
    try:
        interp_field_from_cached = interp_data[indexed_interp_field.astype('int64')]
        interp_field_from_cached.shape = X[subset, subset].shape
    except IndexError:
        print("Use of cached navigation failed, resetting cache")
        if os.path.exists(cache_file):
            os.remove(cache_file)        


    return interp_field_from_cached

def write_GLM_DQP(dqp, x_coord, y_coord, start, end, nadir_lon,
                  platform_ID='G16', orbital_slot='GOES-East', 
                  instrument_ID='GLM-1', outpath='./{dataset_name}',
                  back=None):
    """ Logic here is copied from glmtools.io.imagery.write_goes_imagery. 
                  
    if back is not None, background will be written as a separate variable using the
        variable passed to back.
    """

    back_description = 'GLM Background Image, 1 nm band centered on 777.4 nm, quantized to 16 levels'
                  
    dqp_description = 'GLM Data Quality Product.'\
    ' Upper four bits are the GLM background and the lower four bits are as follows: '\
    +', '.join(dqp_labels)
    
    dataset, scene_id, nominal_resolution = new_goes_imagery_dataset(x_coord,
                                np.flipud(y_coord), nadir_lon)
    global_attrs = get_glm_global_attrs(start, end,
        platform_ID, orbital_slot, instrument_ID, scene_id,
        nominal_resolution, "ABI Mode 3", "DE", "Postprocessed", "TTU"
        )
        
    global_attrs['time_coverage_end'] = str(global_attrs['time_coverage_end']).replace('Z', '.0Z')
    global_attrs['time_coverage_start'] = str(global_attrs['time_coverage_start']).replace('Z', '.0Z')
    dataset = dataset.assign_attrs(**global_attrs)

    # Adding a new variable to the dataset below clears the coord attrs
    # so hold on to them for now.
    xattrs, yattrs = dataset.x.attrs, dataset.y.attrs
    xenc, yenc = dataset.x.encoding, dataset.y.encoding

    # dqp from this script is created with y in the zeroth axis per normal image conventions
    # Transpose here because glmtools expects x in the zeroth axis for reasons I regret.
    image = np.flipud(dqp.T)

    dims = ('y', 'x')
    # 1 is unitless
    img_var = glm_image_to_var(image, 'DQF', dqp_description, '1', dims)#, dtype='u1')
    img_var.encoding['_Unsigned'] = 'true'
    # Assign img_var.encoding['_Unsigned'] = 'true' for AWIPS reasons.
    dataset[img_var.name] = img_var
    
    if back is not None:
        back_img = np.flipud(back.T)
        back_var = glm_image_to_var(back_img, 'background', back_description, '1', dims)#, dtype='u1')
        back_var.encoding['_Unsigned'] = 'true'
        dataset[back_var.name] = back_var
    
    # Restore the cleared coord attrs
    dataset.x.attrs.update(xattrs)
    dataset.y.attrs.update(yattrs)
    dataset.x.encoding.update(xenc)
    dataset.y.encoding.update(yenc)
    
    outfile = outpath.format(start_time=start, end_time=end,
                     dataset_name=dataset.attrs['dataset_name'])
    enclosing_dir = os.path.dirname(outfile)
    if os.path.exists(enclosing_dir) == False:
        os.makedirs(enclosing_dir)

    dataset.to_netcdf(outfile)
    return outfile
        
def dqf_from_nav_background(start, end, lat, lon, 
                            back, back_cal, thresh_fJ,
                            nadir_lon=-75.2,
                            cache_key='dqf_nav_test_cache', cache_path='./', refresh_cache=False,
                            combine_products=True, outpath='./{dataset_name}'):
    """ back is in DN, back_cal is calibrated radiance, 
        thresh_fJis the minimum detectable energy in each pixel.
                            
        combine_products=True packs both products into one 
    """
    

    # ***REPLACE*** with actual thresh_fJ as input
    # thresh_fJ = 8 - 8*back.astype(float)/float(dn_max)
    # below should be fine
    fde = thresh_to_de(thresh_fJ)
    # print(fde.min(), fde.max())

    # ***REPLACE*** with actual boolean mask for is_saturated and near_saturated
    dn_max = 2**14 # 16384
    is_saturated_thresh = dn_max - 2**5 # 2**5=32 
    near_saturated_thresh = dn_max - 2**10
    is_saturated = (back >= is_saturated_thresh)
    near_saturated = ~is_saturated & (back >= near_saturated_thresh)
    
    fde_with_flags = scale_fde(fde)
    fde_with_flags[is_saturated] = at_saturation_val
    fde_with_flags[near_saturated] = near_saturation_val
    
    # Coords of CCD x and y
    x,y = ltg_ellps_lon_lat_to_fixed_grid(lon, lat, nadir_lon, 1)
    
    # Coords of target image. X, Y are 2D meshgrids of y_ctr, x_ctr.
    # bnd variables are the edges instead of the center of the pixels.
    x_bnd, y_bnd, x_ctr, y_ctr, X, Y = get_fixed_grid_coords()
    # print(y_ctr[0], y_ctr[-1])
    
    back_cal_quantized = scale_shift_back(back_cal, shift=combine_products, back_max=dn_max)
    interp_back = interpolate_ccd_to_fixed_grid(back_cal_quantized, x, y, X, Y, 
                                                cache_key, cache_path=cache_path, refresh_cache=refresh_cache)

    interp_dqp = interpolate_ccd_to_fixed_grid(fde_with_flags, x, y, X, Y,
                                               cache_key, cache_path=cache_path, refresh_cache=refresh_cache)

    if combine_products == True:
        # Add the scaled background image into upper bits of the byte
        dqf = np.bitwise_or(interp_back.astype('u1'), interp_dqp.astype('u1')).astype('u1')
        add_back = None
    else: 
        # Separate products each as conventional 1 byte variables.
        dqf = interp_dqp.astype('u1')
        add_back = interp_back.astype('u1')

    outname = write_GLM_DQP(dqf, x_ctr, y_ctr, start, end, nadir_lon, back=add_back, outpath=outpath)
    
    return outname