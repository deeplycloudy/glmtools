import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader as ShapeReader

display_params = {}
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
glm_cmap = get_cmap('viridis')
# glm_cmap._init()
# alphas = np.linspace(1.0, 1.0, glm_cmap.N+3)
# glm_cmap._lut[:,-1] = alphas
# glm_cmap._lut[0,-1] = 0.0

display_params['flash_centroid_density'] = {
    'product_label':"GOES-16 GLM Flash Centroid Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=10),
    'file_tag':'flash_centroid'
}

display_params['flash_extent_density'] = {
    'product_label':"GOES-16 GLM Flash Extent Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=50),
    'file_tag':'flash_extent',
}

display_params['group_centroid_density'] = {
    'product_label':"GOES-16 GLM Group Centroid Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=10),
    'file_tag':'group_centroid'
}

display_params['group_extent_density'] = {
    'product_label':"GOES-16 GLM Group Extent Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=500),
    'file_tag':'group_extent',
}
display_params['event_density']=display_params['group_extent_density']

display_params['total_energy'] = {
    'product_label':"GOES-16 GLM Total Energy (J)",
    'glm_norm':LogNorm(vmin=1e-17, vmax=1e-12),
    'file_tag':'total_energy'
}

display_params[ 'average_flash_area'] = {
    'product_label':"GOES-16 GLM Average Flash Area (km$^2$)",
    'glm_norm':LogNorm(vmin=50, vmax=.5e4),
    'file_tag':'flash_area'
}

display_params['average_group_area'] = {
    'product_label':"GOES-16 GLM Average Group Area (km$^2$)",
    'glm_norm':LogNorm(vmin=50, vmax=.5e4),
    'file_tag':'group_area'
}

label_string = """
{1} (max {0:3.0f})"""

def set_shared_geoaxes(fig):
    mapax =[axi for axi in fig.axes if
            type(axi)==cartopy.mpl.geoaxes.GeoAxesSubplot]
    # for axi in fig.mapax[1:]:
    mapax[0].get_shared_x_axes().join(*mapax)
    mapax[0].get_shared_y_axes().join(*mapax)
    return mapax

def plot_glm(fig, glm_grids, tidx, fields, subplots=(2,3),
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

        ax = fig.add_subplot(subplots[0], subplots[1], fi+1, projection=proj)
        ax.background_patch.set_facecolor(axes_facecolor)

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
        ax.text(limits[0]+.02*(limits[1]-limits[0]), limits[2]+.02*(limits[3]-limits[2]), 
            tidx.isoformat().replace('T', ' ')+' UTC'+
            label_string.format(glm.max(), product_label), 
    #             transform = proj,
            transform = ax.transAxes,
            color=map_color)


        cbars.append((ax,glm_img))

    # Make the colorbar position match the height of the Cartopy axes
    # Have to draw to force layout so that the ax position is correct
    fig.tight_layout()
    fig.canvas.draw()
    for ax, glm_img in cbars:
        posn = ax.get_position()
        height_scale = .025*subplots[0]
        top_edge = [posn.x0, posn.y0+posn.height*(1.0-height_scale),
                    posn.width, posn.height*height_scale]
        cbar_ax = fig.add_axes(top_edge)
        cbar = plt.colorbar(glm_img, orientation='horizontal', cax=cbar_ax,
                    format=LogFormatter(base=2), ticks=LogLocator(base=2))
        cbar.outline.set_edgecolor(axes_facecolor)
        ax.outline_patch.set_edgecolor(axes_facecolor)
        cbar.ax.tick_params(direction='in', color=axes_facecolor, which='both',
                            pad=-14, labelsize=10, labelcolor=axes_facecolor)
    mapax = set_shared_geoaxes(fig)
    return mapax, [ax for (ax,img) in cbars]

