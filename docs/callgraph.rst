Gridding overview
=================

The purpose of this document is to describe the steps in the GLM gridding algorithm,
and where in the code those steps take place. This narrative is focused on the most complete form of the gridding, where the GLM event geometry is reconstructed and remapped onto the ABI fixed grid.

At a high level, processing breaks down into three stages:

1. Preprocessing of the GLM data to account for the event, group, and flash spatial geometry
2. Accumulation of flashes, groups, and events on a target grid with ``lmatools``
3. Creation of NetCDF output files

Call sequence
-------------

The gridding process is driven by a call to ``glmtools.grid.make_grids.grid_GLM_flashes``.
The script ``examples/grid/make_GLM_grids.py`` is a command line script that is useful for
constructing the necessary arguments to call ``grid_GLM_flashes``.

``glmtools.grid.make_grids.grid_GLM_flashes`` drives the gridding process by calling
``glmtools.grid.make_grids.proc_each_grid``, which is called once for each subgrid (so
that processing might be parallelized through tiling).

Each call to ``proc_each_grid`` does the following:

1. creates one ``glmtools.grid.make_grids.GLMGridder`` instance. It creates the lmatools processing pipeline and knows how to read, clip, write the data.
2. creates one ``glmtools.grid.clipping.QuadMeshSubset`` instance for that grid, which is used to find the target grid cells that correspond to each event, group, or flash polygon.
3. creates an ``glmtools.io.glm.GLMDataset`` instance for each data file and sends that through the processing pipeline by calling the ``GLMGridder.process_flashes`` method for each ``GLMDataset`` instance. ``process_flashes`` calls ``glmtools.io.mimic_lma.read_flashes`` to do most of the work.

After reading the data within the target grid (``glmtools.io.glm.GLMDataset.subset_flashes``), bundles of flashes are processed by ``glmtools.io.mimic_lma.read_flash_chunk``. This is the function that drives most of the expensive work. Here's what happens, per the docstring for ``read_flash_chunk``:

0. GLM data are renavigated to the GLM fixed grid and the lightning ellipse removed.
1. GLM event polygons are reconstructed from a corner point lookup table. 
2. The events for each group (flash) are joined into polygons for each group (flash).
3. Split the event polygons by the target grid
4. Split the group polygons by the target grid
5. Split the flash polygons by the target grid
6. For each of the split polygons, calculate its centroid, area, fraction of the original polygon area, and fraction of the target grid cell covered
7. Create numpy arrays with named dtype (i.e., data table with named columns) for (child, parent) pairs at the event, group, and flash levels. See the documentation for mimic_lma_dataset.
8. Send these data to the target, or return them directly if target is None.

Step 0 uses ``glmtools.io.lightning_ellipse.ltg_ellps_lon_lat_to_fixed_grid``. Step 1 uses routines in ``glmtools.io.ccd``. Step 2 uses routines in ``glmtools.io.clipping``
Steps 3-5 use ``glmtools.grid.clipping.QuadMeshPolySlicer.slice``, which also uses ``glmtools.io.clipping``. Step 6 makes use of functions in ``glmtools.grid.split_events``. Step 7 uses routines within ``glmtools.io.mimic_lma.mimic_lma_dataset``. Step 8 uses the ``lmatools`` weighted parent/child point accumulation pipeline set up by ``glmtools.grid.make_grids.GLMGridder``.

Optimization methods
--------------------

The number of events is the primary limit to performance, and there are many events per group and very many event per flash. Each event must be clipped against the underlying grid.

At the moment, the chief optimization methods employed have been to:
1. parallelize clipping into subprocesses
2. send limited bundles of parent/child data at once to lmatools (nonlinear scaling of extent density calculation, I suspect) - i.e., several calls to ``read_flash_chunk`` per GLM file.
3. limit the size of the target grid, saving on total data volume as well as quadrilateral lookup costs. This is not used at the moment. The implementation is buggy near the subgrid edges, and doesn't solve the problem of one very high rate storm within a grid box.

Other options that have been floated are:
1. Cache results of clipping
2. Pre-accumulate the event properties to reduce the total number of events

We initially didn't do any caching, because the assumption was the platform might move. Statistics since have shown that the platform is stable enough to result in much less than 2 km of wobble within one minute.

This should lead to tremendous savings on steps 3 and 6 above for storms with very high event rates.

If we were to add the hash as a column to the xarray dataset, then we could use groupby to aggregate the total energy and count for each unique event location, thereby also reducing the total number of events processed in steps 7 and 8.

Draft plan:
After calculating fixed grid coords with ``ltg_ellps_lon_lat_to_fixed_grid``, assume that events don't move within a minute, and so create a hash of event locations, choosing a suitable rounding amount - perhaps 28 microrad. We then use the hashed exact location and associated clipped polygons to skip unneeded clipping and pre-accumulate, creating a master aggregate event (and associated clipped polygons) for each GLM file, which is then sent to lmatools like normal.


Generation of a call graph graphic
==================================

Below is a sample run that generates a GraphViz call graph, including per-function timing. Requires pycallgraph (installable with pip).

In the resulting .dot file, it helps to comment out (//) the "subgraph" lines to allow
the graph to arrange itself more freely. By default it creates a .png graphic, too, but I also include a version that creates a PDF with the call graph laid out more loosely.

.. code-block:: bash

    pycallgraph -i glmtools.* -i lmatools.* -i stormdrain.* -s -d graphviz -- \
    glmtools/examples/grid/make_GLM_grids.py -o /data/GOES16oklmaMCS/subgrid/ \
    --fixed_grid --subdivide_grid=1 --goes_position test --goes_sector conus \
    --ctr_lon 0.0 --ctr_lat 0.0 --split_events --dx=2.0 --dy=2.0 \
    --start=2017-10-22T04:09:00 --end=2017-10-22T04:10:00 \    
    OR_GLM-L2-LCFA_G16_s20172950409000_e20172950409200_c20172950409231.nc \    
    OR_GLM-L2-LCFA_G16_s20172950409200_e20172950409400_c20172950409431.nc \
    OR_GLM-L2-LCFA_G16_s20172950409400_e20172950410000_c20172950410031.nc > pycallgraph.dot
    
    dot -n1 -Tpdf  pycallgraph.dot  > pycallgraph.pdf
    