import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader as ShapeReader

from glmtools.plot.values import display_params

from matplotlib.cm import get_cmap
glm_cmap = get_cmap('viridis')
# glm_cmap._init()
# alphas = np.linspace(1.0, 1.0, glm_cmap.N+3)
# glm_cmap._lut[:,-1] = alphas
# glm_cmap._lut[0,-1] = 0.0

label_string = """
{1} (max {0})"""
panel_labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

def set_shared_geoaxes(fig):
    mapax =[axi for axi in fig.axes if
            type(axi)==cartopy.mpl.geoaxes.GeoAxesSubplot]
    # for axi in fig.mapax[1:]:
    mapax[0].get_shared_x_axes().join(*mapax)
    mapax[0].get_shared_y_axes().join(*mapax)
    return mapax


def plot_glm_grid(fig, glm_grids, tidx, fields, subplots=(2,3),
             axes_facecolor = (0., 0., 0.), map_color = (.8, .8, .8)):    
    fig.clf()
    glmx = glm_grids.x.data[:]
    glmy = glm_grids.y.data[:]
    proj_var = glm_grids['goes_imager_projection']
    x = glmx * proj_var.perspective_point_height
    y = glmy * proj_var.perspective_point_height    
    glm_xlim = x.min(), x.max()
    glm_ylim = y.min(), y.max()

    country_boundaries = cfeature.NaturalEarthFeature(category='cultural',
                                                      name='admin_0_countries',
                                                      scale='50m', facecolor='none')
    state_boundaries = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lakes',
                                                    scale='50m', facecolor='none')
    globe = ccrs.Globe(semimajor_axis=proj_var.semi_major_axis, semiminor_axis=proj_var.semi_minor_axis)
    proj = ccrs.Geostationary(central_longitude=proj_var.longitude_of_projection_origin,
                              satellite_height=proj_var.perspective_point_height, globe=globe)
    cbars=[]
    
    for fi, f in enumerate(fields):
            
        glm_norm = display_params[f]['glm_norm']
        product_label = display_params[f]['product_label']
        file_tag = display_params[f]['file_tag']
        max_format = display_params[f]['format_string']

        ax = fig.add_subplot(subplots[0], subplots[1], fi+1, projection=proj)
        ax.background_patch.set_facecolor(axes_facecolor)
#         ax.set_aspect('auto', adjustable=None)

        ax.coastlines('10m', color=map_color)
        ax.add_feature(state_boundaries, edgecolor=map_color, linewidth=0.5)
        ax.add_feature(country_boundaries, edgecolor=map_color, linewidth=1.0)
        
        glm = glm_grids[f].sel(time=tidx).data
        #     Use a masked array instead of messing with colormap to get transparency
        #     glm = np.ma.array(glm, mask=(np.isnan(glm)))
        #     glm_alpha = .5 + glm_norm(glm)*0.5
        glm[np.isnan(glm)] = 0
        glm_img = ax.imshow(glm, extent=(x.min(), x.max(), 
                               y.min(), y.max()), origin='upper',
    #                            transform = ccrs.PlateCarree(),
                               cmap=glm_cmap, interpolation='nearest',
                               norm=glm_norm)#, alpha=0.8)

        # Match the GLM grid limits, in fixed grid space
        ax.set_xlim(glm_xlim)
        ax.set_ylim(glm_ylim)

    # Set a lat/lon box directly
#         ax.set_extent([-103, -99.5, 32.0, 38.0])
    
        # limits = ax.axis()
        limits = [0.,1.,0.,1.]
        glm_field_max = max_format.format(glm.max())
        ax.text(limits[0]+.02*(limits[1]-limits[0]), limits[2]+.02*(limits[3]-limits[2]), 
            tidx.isoformat().replace('T', ' ')+' UTC'+
            label_string.format(glm_field_max, product_label), 
            transform = ax.transAxes,
            color=map_color)
        ax.text(limits[0]+.02*(limits[1]-limits[0]),
                limits[3]-.08*(limits[3]-limits[2]),
                panel_labels[fi],
                transform = ax.transAxes,
                color=map_color)

        cbars.append((ax,glm_img))

    # Make the colorbar position match the height of the Cartopy axes
    # Have to draw to force layout so that the ax position is correct
    fig.tight_layout()
    fig.canvas.draw()
    cbar_obj = []
    for ax, glm_img in cbars:
        posn = ax.get_position()
#         internal_left = [posn.x0 + posn.width*.87, posn.y0+posn.height*.05,
#                          0.05, posn.height*.90]
        height_scale = .025*subplots[0]
        top_edge = [posn.x0, posn.y0+posn.height*(1.0-height_scale),
                    posn.width, posn.height*height_scale]
        cbar_ax = fig.add_axes(top_edge)
        cbar = plt.colorbar(glm_img, orientation='horizontal', cax=cbar_ax, #aspect=50, 
                    format=LogFormatter(base=2), ticks=LogLocator(base=2))
        cbar.outline.set_edgecolor(axes_facecolor)
        ax.outline_patch.set_edgecolor(axes_facecolor)
        cbar.ax.tick_params(direction='in', color=axes_facecolor, which='both',
                            pad=-14, labelsize=10, labelcolor=axes_facecolor)
        cbar_obj.append(cbar)
    mapax = set_shared_geoaxes(fig)
    return mapax, cbar_obj
    

