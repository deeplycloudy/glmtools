#!/bin/bash

# usage: make_GLM_grids.py [-h] -o directory --ctr_lat latitude --ctr_lon
#                         longitude --start yyyy-mm-ddThh:mm:ss --end
#                         yyyy-mm-ddThh:mm:ss [--dx km] [--dy km]
#                         [--dt seconds] [--width distance in km]
#                         [--height distance in km]
#                         [filename [filename ...]]

export HOUR=22
export HOUREND=23

echo $HOUR:00:00, $HOUREND:00:00

python make_GLM_grids.py -o /data/LCFA-production/grid_test \
    --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
    --start 2017-05-14T$HOUR:00:00 --end 2017-05-14T$HOUREND:00:00 \
    /data/LCFA-production/GLM-L2-LCFA_G16_s20170514/OR_GLM-L2-LCFA_G16_s2017134$HOUR*.nc

# python make_GLM_grids.py -o /data/LCFA-production/grid_test --lma \
#     --ctr_lat 33.5 --ctr_lon -101.5 --width 800 --height 800 \
#     --start 2017-05-14T$HOUR:00:00 --end 2017-05-14T$HOUREND:00:00 \
#     /data/LCFA-production/LMApost/flashsort/h5_files/2017/May/14/LYLOUT_170514_$HOUR*.h5