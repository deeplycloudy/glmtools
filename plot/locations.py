import matplotlib.pyplot as plt

def plot_flash(glm, some_flash):
    flash_data = glm.get_flashes([some_flash])

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
               flash_data.flash_time_offset_of_first_event.data[0])

    fig = plt.figure()
    ax_ev = fig.add_subplot(111)
    ax_ev.scatter(gr_lon, gr_lat, c=gr_rad, marker='o', s=100, 
                  edgecolor='black', cmap='gray_r') 
    #, vmin=glm.energy_min, vmax=glm.energy_max)
    ax_ev.scatter(ev_lons, ev_lats, c=ev_rad, marker='s', s=16, 
                  edgecolor='black', cmap='gray') 
    #, vmin=glm.energy_min, vmax=glm.energy_max)
    ax_ev.scatter(fl_lon, fl_lat, c='r', marker='x', s=100)
    ax_ev.set_title('GLM Flash #{0}\nfrom {1}\nto {2}'.format(fl_id[0], fl_time[0], fl_time[1]))

    # prevent scientific notation
    ax_ev.get_xaxis().get_major_formatter().set_useOffset(False)
    ax_ev.get_yaxis().get_major_formatter().set_useOffset(False)

    return fig