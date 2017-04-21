""" 

This script converts FGE data provided in the LFCA ASCII format to a
NetCDF format that mimics the production format. It doesn’t have
everything in the production format, but has everything from the ASCII files
that overlaps the production format, and uses the same variable names. 

Author: Eric Bruning, 20 April 2017
License: BSD

"""

import pandas as pd
import xarray as xr


"""
Example of data format:

D Flash #0 Start Time:543398228.069 End Time:543398228.403 Centroid Lat:-4.684 Centroid Lon:-47.148 Energy:1.838e-13 Footprint:715.9 Child_Count:8
	D Group #0 Child #0 Start Time:543398228.069 End Time:543398228.069 Centroid Lat:-4.661 Centroid Lon:-47.120 Energy:9.202e-15 Footprint:179.2 Parent_ID:0 Child_Count:2
		Event #0 Child #0 Time:543398228.069 X_Pixel:1197 Y_Pixel:748 Lat:-4.698 Lon:-47.116 Energy:4.811e-15 Parent_ID:0
		Event #1 Child #1 Time:543398228.069 X_Pixel:1197 Y_Pixel:747 Lat:-4.621 Lon:-47.124 Energy:4.391e-15 Parent_ID:0
	D Group #2 Child #1 Start Time:543398228.083 End Time:543398228.083 Centroid Lat:-4.669 Centroid Lon:-47.144 Energy:1.792e-14 Footprint:268.6 Parent_ID:0 Child_Count:3
		Event #4 Child #0 Time:543398228.083 X_Pixel:1197 Y_Pixel:748 Lat:-4.698 Lon:-47.116 Energy:6.440e-15 Parent_ID:2
		Event #5 Child #1 Time:543398228.083 X_Pixel:1197 Y_Pixel:747 Lat:-4.621 Lon:-47.124 Energy:6.664e-15 Parent_ID:2
		Event #6 Child #2 Time:543398228.083 X_Pixel:1196 Y_Pixel:748 Lat:-4.697 Lon:-47.210 Energy:4.817e-15 Parent_ID:2
"""

flash_cols = (2, 4, 6, 8, 10, 11, 12, 13)
flash_col_names = ('flash_id', 
                   'flash_time_offset_of_first_event', 'flash_time_offset_of_last_event', 
                   'flash_lat', 'flash_lon', 'flash_energy', 'flash_area', 
                   'flash_child_group_count')
group_cols = (2, 6, 10, 12, 13, 14, 15, 16)
group_col_names = ('group_id', 
                   'group_time_offset', 
                   'group_lat', 'group_lon', 'group_energy', 'group_area', 
                   'group_parent_flash_id', 'group_child_event_count')

event_cols = (1, 4, 5, 6, 7, 8, 9, 10)
event_col_names = ('event_id', 
                   'event_time_offset', 'x_pixel', 'y_pixel',
                   'event_lat', 'event_lon', 'event_energy',
                   'event_parent_group_id',)

# Columns treated as coordinate data in the NetCDF file                   
nc_coords = [
'event_id',
'event_time_offset',                  
'event_lat',                          
'event_lon',                          
'event_parent_group_id',              
'group_id',                           
'group_time_offset',                  
'group_lat',                          
'group_lon',                          
'group_parent_flash_id',              
'flash_id',                           
'flash_time_offset_of_first_event',   
'flash_time_offset_of_last_event',    
'flash_lat',                          
'flash_lon',
]
                   

def gen_flash_data(filename):
    """
        0   1    2    3       4              5      6                  7       8          9        10              11               12           13
        D Flash #0 Start Time:543398228.069 End Time:543398228.403 Centroid Lat:-4.684 Centroid Lon:-47.148 Energy:1.838e-13 Footprint:715.9 Child_Count:8    
    """
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("D Flash"):
                cols = line.split()
                out = (int(cols[2][1:]),
                       float(cols[4].split(':')[1]),
                       float(cols[6].split(':')[1]),
                       float(cols[8].split(':')[1]),
                       float(cols[10].split(':')[1]),
                       float(cols[11].split(':')[1]),
                       float(cols[12].split(':')[1]),
                       int(cols[13].split(':')[1]),
                       )                      
                yield out

def gen_group_data(filename):
    """
        0   1    2   3    4   5        6              7         8              9      10          11         12         13               14              15          16          
    	D Group #0 Child #0 Start Time:543398228.069 End Time:543398228.069 Centroid Lat:-4.661 Centroid Lon:-47.120 Energy:9.202e-15 Footprint:179.2 Parent_ID:0 Child_Count:2
    """
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("	D Group"):
                cols = line.split()
                out = (int(cols[2][1:]),
                       float(cols[6].split(':')[1]),
                       float(cols[10].split(':')[1]),
                       float(cols[12].split(':')[1]),
                       float(cols[13].split(':')[1]),
                       float(cols[14].split(':')[1]),
                       int(cols[15].split(':')[1]),
                       int(cols[16].split(':')[1]),
                       )                      
                yield out

def gen_event_data(filename):
    """
               0   1  2    3          4                 5          6          7          8           9              10    
    		Event #0 Child #0 Time:543398228.069 X_Pixel:1197 Y_Pixel:748 Lat:-4.698 Lon:-47.116 Energy:4.811e-15 Parent_ID:0
    """
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("		Event"):
                cols = line.split()
                out = (int(cols[1][1:]),
                       float(cols[4].split(':')[1]),
                       int(cols[5].split(':')[1]),
                       int(cols[6].split(':')[1]),
                       float(cols[7].split(':')[1]),
                       float(cols[8].split(':')[1]),
                       float(cols[9].split(':')[1]),
                       int(cols[10].split(':')[1]),
                       )                      
                yield out

                
def parse_LCFA_ASCII(filename):
    """
    Read ASCII Flash, Group, Event data in filename and return 
    pandas dataframes (flashes, groups, events)
    """
    # Right now this reads through the data file three times. Could use a coroutine approach to read file once
    # if performance starts to take a hit.
    
    flashes = pd.DataFrame.from_records(gen_flash_data(filename), columns=flash_col_names)#, index='flash_id')
    groups = pd.DataFrame.from_records(gen_group_data(filename), columns=group_col_names)#, index='group_id')
    events = pd.DataFrame.from_records(gen_event_data(filename), columns=event_col_names)#, index='event_id')

    # Dimension names used in the NetCDF file
    flashes.index.name = 'number_of_flashes'
    groups.index.name = 'number_of_groups'
    events.index.name = 'number_of_events'
    
    return flashes, groups, events
    
def merge_dataframes_to_xarray(flashes, groups, events):
    evx = events.to_xarray()
    flx = flashes.to_xarray()
    grx = groups.to_xarray()
    egf = evx.merge(grx).merge(flx)
    egf.set_coords(nc_coords, inplace=True)
    return egf

def main(infile, outfile):
    fl, gr, ev = parse_LCFA_ASCII(infile)
    egf = merge_dataframes_to_xarray(fl, gr, ev)
    egf.to_netcdf(outfile)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_lcfa.py input_filename output_filename")
    else:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        main(infile, outfile)
