.. glmtools documentation master file, created by
   sphinx-quickstart on Tue Jul 25 11:27:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for glmtools
==========================

.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
   callgraph

Installation
============

glmtools requires ``Python >= 3.5``, ``numpy``, ``scipy``, ``netCDF4``, and
``xarray >= 0.9.7``. That version of ``xarray`` is the first to support automatic
decoding of the ``_Unsigned`` integer data that is used throughout the GLM
NetCDF files. ``matplotlib`` is used for plotting. Gridding and flash statistics
are built on top of ``lmatools``.

glmtools includes CCD pixel shape information in fixed grid coordinates, so that
``glmtools`` can regrid the GLM event detections onto an arbitrary target grid using
fractional pixel coverage. Doing so requires the ``pyclipper`` wrapper for the freeware
`Clipper <http://www.angusj.com/delphi/clipper.php>`_ c++ library.

Step by step instructions
-------------------------

The instructions below assume the Anaconda Python distribution (``miniconda`` is
fine), and the availability of the ``conda`` package manager.

.. code-block:: bash

   git clone https://github.com/deeplycloudy/glmtools.git
   cd glmtools
   conda env create -f environment.yml
   conda activate glmval
   pip install -e .

If you want to use the example notebooks also run
"conda install -c conda-forge ipython matplotlib"

Using glmtools
==============

Ensure you are in the anaconda environment created earlier
----------------------------------------------------------
Be sure to activate the ``conda`` environment that contains glmtools.

If you are using the environment created by environment.yml, this environment will be named  ``glmval``:

.. code-block:: bash

   conda activate glmval


Get some GLM data, and check it for sanity
------------------------------------------

The GLM Level 2 data can be obtained from a GOES Rebroadcast feed, an LDM or THREDDS
service, the Amazon S3 bucket that is part of the NOAA Big Data Project or perhaps other
sources. These files are in NetCDF 4 format, and begin with ``OR_GLM-L2-LCFA``.

Three sample data files are included in ``glmtools/test/data``.

Early preliminary and non-operational data had some internal inconsistencies that prevent
further use with glmtools. Before undertaking further processing, it is recommended that
the files be checked for sanity using the ``examples/check_glm_data.py`` script, as shown
below. The script contains some documentation about what is checked.

.. code-block:: bash

   python check_glm_data.py /path/to/GLM/data/OR_GLM-L2-LCFA_*.nc

If a file is shown to have an error, change that file's extension to
``.nc.bad``, which will make it easy to ignore with a wildcard match in
subsequent steps.

Grid the GLM data
-----------------

The script ``examples/grid/make_GLM_grids.py`` is a command line utility; run with ``--help`` for usage.

For instance, the following command, using the included sample data will grid
one minute of data (3 GLM files) on the ABI fixed grid in the CONUS sector at 2
km resolution. These images will overlay precisely on the ABI cloud tops, and
will have parallax with respect to ground for all the same reasons ABI does.
Output will be placed in the current directory in a new 2018/Jul/02 directory
created by default.

.. code-block:: bash

    python make_GLM_grids.py
    --fixed_grid --split_events \
    --goes_position east --goes_sector conus \
    --dx=2.0 --dy=2.0 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \


If you don't need the whole conus sector, you can instead plot on a mesoscale domain centered at an arbitrary point. This will be about 1000 x 1000 km, same as the ABI fixed grid meso sector.

.. code-block:: bash

    python make_GLM_grids.py
    --fixed_grid --split_events \
    --goes_position east --goes_sector meso \
    --dx=2.0 --dy=2.0 \
    --ctr_lon=-101.5 --ctr_lat=33.5 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \

Finally, if you want a fully custom grid size, you can omit the ``--goes_sector`` argument and specify a width and height in kilometers.

.. code-block:: bash

    python make_GLM_grids.py
    --fixed_grid --split_events \
    --goes_position east \
    --dx=2.0 --dy=2.0 --width="1000.0" --height="500.0" \
    --ctr_lon=0.0 --ctr_lat=0.0 \
    OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc \
    OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc \
    OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc \


The notebook ``examples/plot_glm_test_data.ipynb`` runs the meso sector example in a temporary directory, and also shows how to plot the resulting grids on a map. In the examples above, the start and end time of the grids was inferred from the filenames, but the notebook also shows how to grid 1 min of data to a file containing 5, 1-min frames, all but one of which will be empty.

Calculate time series flash rate data
-------------------------------------

The script ``examples/glm-lasso-stats.py`` is a command line utility; run with
--help for usage. There is an equivalent script in ``lmatools`` with very
similar arguments. This script requires a cell lasso file, which may be as
simple as a bounding box, or as elaborate as a time-evolving pattern.

An example lasso file is found in ``examples/lasso-WTLMA-50km-2017Jul05.txt``. This
simple rectangular bounding box is centered on the West Texas LMA and is valid for a few
hours on 5 July 2017. Both the valid time and the coordinates can be edited directly to
change to a different day or box. The first and last vertices of the bounding box (any
polygon is valid) must be repeated to close the polygon.

Suggestions for automating
--------------------------

The script ``examples/glm_lma_param_space.sh`` shows how to combine the above
pieces into a large parameter space study, applied to both GLM and LMA data on
the same grid and for the same bounding box.

To run the script, it is recommended to copy it and your bounding box file
to an analysis directory. Then, edit the export lines at the beginning of
the script to point your files, dates, times, grid specification, etc.
Output from the script is saved to the ``GLMSORTGRID`` and ``LMASORTGRID``
directories you have specified.

A wealth of time series statstics will be calculated, and saved to .csv files.
Of particular interest are:

- ``flash_stats.csv``
- ``grids_flash_extent/flash_extent_{date}.csv``

The first contains flash rate, average flash size, and other data calculated from the raw (ungridded) flashd data. The second contains statstics of the
population of grid cells, such as the max, min, and various percentiles.

Reference plots of most of these data are also created.

- ``flash_stats_{start}_{end}.pdf``
- ``grids_flash_extent/*.png``


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

