#!/bin/bash

# usage: make_GLM_grids.py [-h] -o directory --ctr_lat latitude --ctr_lon
#                         longitude --start yyyy-mm-ddThh:mm:ss --end
#                         yyyy-mm-ddThh:mm:ss [--dx km] [--dy km]
#                         [--dt seconds] [--width distance in km]
#                         [--height distance in km]
#                         [filename [filename ...]]

python make_GLM_grids.py -o /data/LCFA-production/grid_test --ctr_lat 33.5 --ctr_lon -101.5 --start 2017-04-29T08:00:00 --end 2017-04-29T09:00:00 /data/LCFA-production/GLM-L2-LCFA_G16_s20170429/OR_GLM-L2-LCFA_G16_s201711908*.nc