export MPLBACKEND="Agg"

export GLMGRIDSCRIPT=~/sources/glmtools/examples/grid/make_GLM_grids.py
export GLMLASSOSCRIPT=~/sources/glmtools/examples/glm-lasso-stats.py
export LMALASSOSCRIPT=~/sources/lmatools/examples/lasso/cell-lasso-stats.py

export GLMSORTGRID=./GLM/
export LMASORTGRID=./LMA/

export LASSO=lasso-WTLMA-50km-2017Jul05.txt

export CTRLAT=33.5
export CTRLON=-101.5
export WIDTH=800

export GLMFILES=/archive/GLM/GLM-L2-LCFA_G16/2017/Jul/05/OR_GLM-L2-LCFA_G16_s20171860[1-4]*.nc
export LMAFILES=/archive/GLM/LMA/h5_files/2017/Jul/05/LYLOUT_170705_0[1-4]*.h5

echo "Processing LMA flashes to grid"
python $GLMGRIDSCRIPT \
    -o $LMASORTGRID/grid_files/ \
    --nevents 10 \
    --ctr_lat $CTRLAT --ctr_lon $CTRLON --width $WIDTH --height $WIDTH \
    --lma $LMAFILES

echo "Processing GLM flashes to grid"
for GLMEVENTS in 1 #2 4
do
    python $GLMGRIDSCRIPT \
        -o $GLMSORTGRID/minevent$GLMEVENTS/grid_files/ \
        --nevents $GLMEVENTS \
        --ctr_lat $CTRLAT --ctr_lon $CTRLON --width $WIDTH --height $WIDTH \
        $GLMFILES &
done

wait

for LMAAREA in 1 #4 16 32 64 256 1024
# all flashes with greater than this area
do
    for LMAALT in 0 #4 6 8 10 12 16
    # all flashes whose centroid is greater than each of these altitudes
    do
        echo "Processing LMA flashes for area$LMAAREA_energy$LMAENERGY"
        python $LMALASSOSCRIPT -l $LASSO -s $LMASORTGRID --skipspectra \
            --minarea $LMAAREA --minalt $LMAALT \
            -o LMA_area$LMAAREA\_ctralt$LMAALT \
            $LMAFILES &
    done
    wait
done

for GLMEVENTS in 1 #2 4
# Use GLM grids that have been filtered by min events before producing
do
    for GLMAREA in 0 #128 256 512 1024 4096 16384
    # all flashes greater than this area
    do
        for GLMENERGY in 0 #"1.0e-14" "5.0e-14" "1.0e-13" "5.0e-13" "1.0e-12"
        # all flashes larger than this energy/radiance
        do
            python $GLMLASSOSCRIPT --skip3d -l $LASSO --skipspectra \
                -s $GLMSORTGRID/minevent$GLMEVENTS/ \
                --nevents $GLMEVENTS --minarea $GLMAREA --minenergy $GLMENERGY \
                -o GLM_events$GLMEVENTS\_area$GLMAREA\_energy$GLMENERGY \
                $GLMFILES &
        done
        wait
    done
done