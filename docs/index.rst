.. glmtools documentation master file, created by
   sphinx-quickstart on Tue Jul 25 11:27:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for glmtools
==========================

Installation Requirements
=========================

glmtools requires ``Python >= 3.5``, ``numpy``, ``scipy``, ``netCDF4``, and
``xarray >= 0.97``. That version of ``xarray`` is the first to support automatic
decoding of the ``_Unsigned`` integer data that is used throughout the GLM
NetCDF files. ``matplotlib`` is used for plotting. Gridding and flash statistics
are built on top of ``lmatools``.

Step by step instructions
------------------------- 

The instructions below assume the Anaconda Python distribution (`miniconda` is
fine), and the availability of the ``conda`` package manager.

Until development on ``xarray`` and ``lmatools`` stabilizes, it is recommended
to install those packages from source. From some working directory (e.g.,
``~/sources``):

.. code-block:: bash

   git clone https://github.com/deeplycloudy/lmatools.git
   git clone https://github.com/deeplycloudy/stormdrain.git
   git clone https://github.com/deeplycloudy/glmtools.git
   git clone https://github.com/pydata/xarray.git
   cd glmtools
   conda env create -f environment.yml
   source activate glmval
   cd ../lmatools
   python setup.py install
   cd ../stormdrain
   python setup.py install
   cd ../xarray
   pip install -e ./
   cd ../glmtools
   pip install -e ./
   

Recommended Analyses
====================

Ensure you are in the anaconda environment created earlier
----------------------------------------------------------
Be sure to activate the ``conda`` environment that contains glmtools.

If you are using the environment created by environment.yml, this environment will be named  ``glmval``:

.. code-block:: bash

   source activate glmval


Get some GLM data, and check it for sanity
------------------------------------------

The GLM Level 2 data that can be obtained from, for example, a GOES Rebroadcast
feed, are preliminary and non-operational. These files are in NetCDF 4 format,
and begin with ``OR_GLM-L2-LCFA_G16``. Before undertaking further processing,
is recommended that the files be checked for sanity using the
``examples/check_glm_data.py`` script, as shown below. The script contains more
documentation about what is checked.

.. code-block:: bash

   python check_glm_data.py /path/to/GLM/data/OR_GLM-L2-LCFA_*.nc

If a file is shown to have an error, change that file's extension to
``.nc.bad``, which will make it easy to ignore with a wildcard match in
subsequent steps.

Grid the GLM data
-----------------

The script ``examples/grid/make_GLM_grids.py`` is a command line utility; run with ``--help`` for usage. The same script can be used to grid LMA data on the same grid by adding the ``--lma`` flag. This step requires LMA HDF5 files containing flash-sorted data. This step produces NetCDF grids in regular lat/lon space.

Calculate time series flash rate data 
------------------------------------- 

The script ``examples/glm-lasso-stats.py`` is a command line utility; run with
--help for usage. There is an equivalent script in ``lmatools`` with very
similar arguments. This script requires a cell lasso file, which may be as
simple as a bounding box, or as elaborate as a time-evolving pattern.

An example lasso file is found in ``examples/lasso-WTLMA-50km-2017Jul05.txt``.
This simple rectangular bounding box is centered on the West Texas LMA and is
valid for a few hours on 5 July 2017. Both the valid time and the coordinates
can be edited directly. The first and last vertices of the bounding box (any polygon is valid) must be repeated to close the polygon.
 
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

