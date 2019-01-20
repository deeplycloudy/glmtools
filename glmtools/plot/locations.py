import matplotlib.pyplot as plt

def plot_flash(glm, flash_id, ax=None, proj=None):
    flash_data = glm.get_flashes([flash_id])

    ev_parent = flash_data.event_parent_group_id  
    gr_parent = flash_data.group_parent_flash_id


    gr_id = flash_data.group_id.data
    fl_id = flash_data.flash_id.data

    ev_lats = flash_data.event_lat.data
    ev_lons = flash_data.event_lon.data
#     ev_lats, ev_lons = fix_event_locations(flash_data.event_lat, flash_data.event_lon, is_xarray=True)
    ev_time = flash_data.event_time_offset.data
    ev_rad = flash_data.event_energy.data
    gr_lat = flash_data.group_lat.data
    gr_lon = flash_data.group_lon.data
    gr_rad = flash_data.group_energy.data
    fl_lat = flash_data.flash_lat.data
    fl_lon = flash_data.flash_lon.data
    fl_rad = flash_data.flash_energy.data
    fl_time = (flash_data.flash_time_offset_of_first_event.data[0], 
               flash_data.flash_time_offset_of_last_event.data[0])

    if ax is None:
        fig = plt.figure()
        ax_ev = fig.add_subplot(111)
    else:
        ax_ev = ax
    
    gr_kwargs = dict(c=gr_rad, marker='o', s=100, 
                     edgecolor='black', cmap='gray_r')
    #, vmin=glm.energy_min, vmax=glm.energy_max)
    if proj: gr_kwargs['transform'] = proj
    ax_ev.scatter(gr_lon, gr_lat, **gr_kwargs) 

    ev_kwargs = dict(c=ev_rad, marker='s', s=16, 
                     edgecolor='black', cmap='gray') 
    #, vmin=glm.energy_min, vmax=glm.energy_max)
    if proj: ev_kwargs['transform'] = proj
    ax_ev.scatter(ev_lons, ev_lats, **ev_kwargs)
    
    fl_kwargs = dict(c='r', marker='x', s=100)
    if proj: fl_kwargs['transform'] = proj
    ax_ev.scatter(fl_lon, fl_lat, **fl_kwargs)
    ax_ev.set_title('GLM Flash #{0}\nfrom {1}\nto {2}'.format(
        fl_id[0], fl_time[0], fl_time[1]))

    # prevent scientific notation
    ax_ev.get_xaxis().get_major_formatter().set_useOffset(False)
    ax_ev.get_yaxis().get_major_formatter().set_useOffset(False)

    return ax_ev


if __name__ == '__main__':
    from glmtools.plot.locations import plot_flash    
    from glmtools.io.glm import GLMDataset
    glm = GLMDataset('/data/LCFA-production/OR_GLM-L2-LCFA_G16_s20171161230400_e20171161231000_c20171161231027.nc')
    plot_flash(glm, 6666)