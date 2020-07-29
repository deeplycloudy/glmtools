## Plotting script for GLM grids

**BETA - these instructions may be outdated.**

### Requirements

- glmtools
- cartopy

### Usage

1. Copy `plot_glm_file.py` and `plots.py` to a working directory
2. Create a directory in which to save figures, e.g., `./loop`
3. Run the `plot_glm_file.py` script, as shown below

```bash
python plot_glm_file.py -o ./loop \
  ./GLM_GRID/2018/Oct/28/OR_GLM-L2-GLMC-M3_G16_s2018301201*.nc
```

This will create a directory in 2018/Oct/28 and .png files for each of the
minutes in the GLM dataset.

### Embedding in a notebook

```python
from plots import plot_glm

fields_6panel = ['flash_extent_density', 'average_flash_area','total_energy', 
                 'group_extent_density', 'average_group_area', 'group_centroid_density']



def plot(w, fig=None, time_widget=None, field_widget=None, subplots=(2,3), fields=None):
    t = pd.to_datetime(time_widget.value)
    n_subplots = subplots[0] * subplots[1]
    if fields is None:
        if n_subplots == 1:
            fields = [field_widget.value]
        else:
            fields = fields_6panel[0:n_subplots]

    plot_glm(fig, glm_grids, t, fields, subplots=subplots)

fig = plt.figure(figsize=(18,12))


from ipywidgets import widgets

time_options = [str(t0) for t0 in glm_grids.time.to_series()]
time_options.sort()
field_options = list(glm_grids.variables.keys())
for v in ['x', 'y', 'time', 'goes_imager_projection', 'DQF']: field_options.remove(v)
field_dropdown = widgets.Dropdown(options=field_options)
time_slider = widgets.SelectionSlider(options=time_options)
glm_select = widgets.HBox([field_dropdown, time_slider])

from functools import partial
plot = partial(plot, fig=fig, field_widget=field_dropdown, time_widget=time_slider)
time_slider.observe(plot)
field_dropdown.observe(plot)

display(glm_select)
time_slider.value = time_slider.options[3]

```