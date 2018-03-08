# glmtools

GOES Geostationary Lightning Mapper Tools

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

## Some common tasks

### Create gridded NetCDF imagery

Use the script in `examples/grid/make_GLM_grids.py`. For instance, the
following command will grid one minute of data (3 GLM files) on the ABI fixed
grid in the CONUS sector at 10 km resolution.

Note that these grids will either have gaps or will double-count events along
GLM pixel borders, because there is no one grid resolution which exactly
matches the GLM pixel size as it varies with earth distortion over the field
of view.

```bash 
python make_GLM_grids.py -o /path/to/output/ --fixed_grid --goes_position east \
--goes_sector conus --ctr_lon 0.0 --ctr_lat 0.0 --dx=10.0 --dy=10.0 \
--start=2018-01-04T05:37:00 --end=2018-01-04T05:38:00 \
OR_GLM-L2-LCFA_G16_s20180040537000_e20180040537200_c20180040537226.nc \
OR_GLM-L2-LCFA_G16_s20180040537200_e20180040537400_c20180040537419.nc \
OR_GLM-L2-LCFA_G16_s20180040537400_e20180040538000_c20180040538022.nc \
```

To start with, look at the flash extent density and total energy grids.

`ctr_lon` and `ctr_lat` aren't used, but are required anyway. Fixing this would
make a nice first contribution!

### Interactively plot raw flash data

See the examples folder. `plot_glm_test_data.ipynb` is a good place to start.

### Reduce the dataset to a few flashes

`smaller_dataset = glmtools.io.glm.GLMDataset.get_flashes[flash_id_list]`

### Filter out flashes with single-event groups

```python
from glmtools.test.common import get_test_dataset
glm, flash_ids = get_test_dataset()
fl_idx = glm.dataset['flash_child_event_count'] > 2
flash_ids = glm.dataset[{glm.fl_dim: fl_idx}].flash_id.data
smaller_dataset = glm.get_flashes(flash_ids)
print(smaller_dataset)
```

The same logic can be used to find big flashes. For some common needs, like
finding flashes in a bounding box or with minimum number of events or groups,
see `glm.subset_flashes(...)`

### Filter out a lat-lon subset of flashes

See `glmtools.io.glm.GLMDataset.lonlat_subset(...)`



