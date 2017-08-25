#!/bin/bash

# usage: make_GLM_grids.py [-h] -o directory --ctr_lat latitude --ctr_lon
#                         longitude --start yyyy-mm-ddThh:mm:ss --end
#                         yyyy-mm-ddThh:mm:ss [--dx km] [--dy km]
#                         [--dt seconds] [--width distance in km]
#                         [--height distance in km]
#                         [filename [filename ...]]

export HOUR=05
export HOUREND=06

# python make_GLM_grids.py -o /data/LCFA-production/grid_test_100groups \
#     --ngroups 100 \
#     --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
#     --start 2017-05-15T$HOUR:00:00 --end 2017-05-15T$HOUREND:00:00 \
#     /data/LCFA-production/GLM-L2-LCFA_G16_s20170514/OR_GLM-L2-LCFA_G16_s2017135$HOUR*.nc

# python make_GLM_grids.py -o /data/LCFA-production/grid_test_2groups \
#     --ngroups 2 \
#     --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
#     --start 2017-05-15T$HOUR:00:00 --end 2017-05-15T$HOUREND:00:00 \
#     /data/LCFA-production/GLM-L2-LCFA_G16_s20170514/OR_GLM-L2-LCFA_G16_s2017135$HOUR*.nc

python make_GLM_grids.py -o /data/LCFA-production/grid_test_3groups \
    --ngroups 3 \
    --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
    /data/LCFA-production/GLM-L2-LCFA_G16_s20170514/OR_GLM-L2-LCFA_G16_s2017135$HOUR*.nc


# python make_GLM_grids.py -o /data/LCFA-production/grid_test --lma \
#     --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
#     --start 2017-05-15T$HOUR:00:00 --end 2017-05-15T$HOUREND:00:00 \
#     /data/LCFA-production/LMApost/flashsort/h5_files/2017/May/15/LYLOUT_170515_$HOUR*.h5