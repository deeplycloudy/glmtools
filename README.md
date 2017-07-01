# glmtools

GOES Geostationary Lightning Mapper Tools

Requirements:
- Python 3.5+
- xarray

Compatible data:
- NetCDF format Level 2 (L2) data from the GOES Re-Broadcast (GRB) feed.

glmtools automatically reconstitutes the parent-child relationships implicit  in the L2 GLM data and adds traversal information to the dataset:
- calculating the parent flash id for each event
- calculating the number of groups and events in each flash
- calculating the number of events in each group

xarray's dimension-aware indexing lets you quickly reduce the dataset to 
flashes of interest, as described below.

## Some common tasks


### Interactively plot raw flash data and create flash extent density grids

See the examples folder. `plot_glm_test_data.ipynb` is a good place to start.

### Reduce the dataset to a few flashes

`smaller_dataset = glmtools.io.glm.GLMDataset.get_flashes[flash_id_list]`

### Filter out flashes with single-event groups

`from glmtools.test.common import get_test_dataset`
`glm, flash_ids = get_test_dataset()`
`glmtools.io.glm.GLMDataset.dataset['flash_child_event_count'] > 2`
`flash_ids = glm.dataset[{glm.fl_dim: fl_idx}].flash_id.data`
`smaller_dataset = glmtools.io.glm.GLMDataset.get_flashes[flash_ids]`
`print(smaller_dataset)`
The same logic can be used to find big flashes.

### Filter out a lat-lon subset of flashes

See `glmtools.io.glm.GLMDataset.lonlat_subset(...)`



