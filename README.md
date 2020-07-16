# glmtools

GOES Geostationary Lightning Mapper Tools

[![DOI](https://zenodo.org/badge/71308485.svg)](https://zenodo.org/badge/latestdoi/71308485)

## Installation
glmtools requires Python 3.5+ and provides a conda `environment.yml` for the key dependencies.

See the documentation in `docs/index.rst` for complete installation instructions.

## Description
Compatible data:
- NetCDF format Level 2 data, as described in the [Product Definition and Users Guide](https://www.goes-r.gov/resources/docs.html)

glmtools automatically reconstitutes the parent-child relationships implicit in
the L2 GLM data and adds traversal information to the dataset:

- calculating the parent flash id for each event
- calculating the number of groups and events in each flash
- calculating the number of events in each group

xarray's dimension-aware indexing lets you quickly reduce the dataset to 
flashes of interest, as described below.

glmtools can restore the GLM event geometry using a built-in corner-point lookup table,
which allows for gridding of the imagery at finer resolutions that accurately represent
the full footprint of each event, group, and flash.

The methods are described in Bruning (2019):

-  Bruning, E., Tillier, C. E., Edgington, S. F., Rudlosky, S. D.,
Zajic, J., Gravelle, C., et al. (2019). Meteorological imagery for
the geostationary lightning mapper. Journal of Geophysical Research:
Atmospheres, 2019; 124: 14285 14309. https://doi.org/10.1029/2019JD030874

## Some common tasks

### Create gridded NetCDF imagery

Use the script in `examples/grid/make_GLM_grids.py`. See the documentation in `docs/index.rst` for complete instructions and example commands.

### Interactively plot raw flash data

See the examples folder. `basic_read_plot.ipynb` is a good place to start.

### Reduce the dataset to a few flashes

```python
from glmtools.io.glm import GLMDataset
filename = 'OR_GLM-L2-LCFA_G16_s20180040537000_e20180040537200_c20180040537226.nc'
glm =  GLMDataset(filename)
flash_id_list = glm.dataset.flash_id[20:30]
smaller_dataset = glm.get_flashes(flash_id_list)
```

### Filter out flashes geographically or by events/groups per flash

See `glmtools.io.glm.GLMDataset.subset_flashes`.

The logic implemented above is pretty simple, and below shows how to adapt it to find large flashes.

```python
from glmtools.io.glm import GLMDataset
filename = 'OR_GLM-L2-LCFA_G16_s20180040537000_e20180040537200_c20180040537226.nc'
glm =  GLMDataset(filename)
fl_idx = glm.dataset['flash_area'] > 2000
flash_ids = glm.dataset[{glm.fl_dim: fl_idx}].flash_id.data
smaller_dataset = glm.get_flashes(flash_ids)
print(smaller_dataset)
```

