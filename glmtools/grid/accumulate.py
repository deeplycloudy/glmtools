import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import numpy as np

from lmatools.stream.subset import coroutine

@coroutine
def select_dataset(target, use_event_data=True):
    """ Check for data and send either events or flashes,
        depending on the kwarg *events*.
    """
    try:
        while True:
            events, flashes = (yield)
            if use_event_data:
                if len(events) > 0:
                    target.send(events)
            else:
                if len(flashes) > 0:
                    target.send(flashes)
    except GeneratorExit:
        pass


@coroutine
def accumulate_var_on_grid_direct_idx(grid, var_name, x_idx, y_idx, out=None):
    """
    coroutine that awaits a pandas dataframe or xarray dataset and accumulates
        it on *grid*.
    grid - a 2D array that is indexed along its first and second dimensions by
        xi = dataset[x_idx] and yi = dataset[y_idx], respectively.
    var_name - the 1D array to accumulate, i.e., dataset[var_name]
    x_idx, y_idx: names of 1D arrays in in the dataset giving the indices into
        grid onto which dataset[var_name] is accumulated.
    
    This function assumes that pairs of x_idx and y_idx only occur once, and
    that logic prior to this function has eliminated and pre-accumulated all
    but one of the values at each grid cell.
    
    This is unlike the earlier accumulation functions in lmatools, which used
    histogramdd to weight and sum the data. This function expects the weighting
    and accumulation to have been done prior to this through manual
    multiplacation and a groupby. Then, this function is a sort of dumb
    accumulator.
    
    To mimic the functionality of histogramdd, we need to groupby and sum
    over the x_idx and y_idx variables. This will give us a single accumulated
    value that can be placed onto the grid by assigning grid[xi, yi]
    """
    if out == None:
        out = {}
    
    try:
        while True:
            dataset = (yield)
            xi, yi = dataset[x_idx], dataset[y_idx]
            grid[yi, xi] += dataset[var_name].astype(grid.dtype)
            del xi, yi, dataset
    except GeneratorExit:
        out['out'] = grid
        
        
@coroutine
def accumulate_minvar_on_grid_direct_idx(grid, var_name, x_idx, y_idx, out=None):
    """
    coroutine that awaits a pandas dataframe or xarray dataset and accumulates
        it on *grid*.
    grid - a 2D array that is indexed along its first and second dimensions by
        yi = dataset[y_idx] and xi = dataset[x_idx], respectively.
    var_name - the 1D array to accumulate, i.e., dataset[var_name]
    x_idx, y_idx: names of 1D arrays in in the dataset giving the indices into
        grid onto which dataset[var_name] is accumulated.

    This function assumes that pairs of x_idx and y_idx only occur once, and
    that logic prior to this function has eliminated and pre-accumulated all
    but one of the values at each grid cell.

    This is unlike the earlier accumulation functions in lmatools, which used
    histogramdd to weight and sum the data. This function expects the weighting
    and accumulation to have been done prior to this through manual
    multiplacation and a groupby. Then, this function is a sort of dumb
    accumulator.

    To mimic the functionality of histogramdd, we need to groupby and sum
    over the x_idx and y_idx variables. This will give us a single accumulated
    value that can be placed onto the grid by assigning grid[xi, yi]
    """
    if out == None:
        out = {}

    try:
        while True:
            dataset = (yield)
            xi, yi = dataset[x_idx], dataset[y_idx]
            new_min = dataset[var_name].astype(grid.dtype)
            
            # two possibliities: grid is zero, grid is nonzero.
            prev_values = grid[yi, xi]
            gridnonzero = (prev_values > 0)
            gridzero = ~gridnonzero # np.isclose(grid, 0)
            # where grid is zero, just use the new value, otherwise take the
            # min of both
            grid[yi[gridzero], xi[gridzero]] = new_min[gridzero]
            grid[yi[gridnonzero], xi[gridnonzero]] = np.minimum(
                new_min[gridnonzero], prev_values[gridnonzero]
                )
            del xi, yi, dataset, gridzero, gridnonzero, prev_values, new_min
    except GeneratorExit:
        out['out'] = grid