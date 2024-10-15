#!/usr/bin/env python
# coding: utf-8

# Before running this notebook, it's helpful to 
# 
# `conda install -c conda-forge nb_conda_kernels`
# 
# `conda install -c conda-forge ipywidgets`
# 
# and set the kernel to the conda environment in which you installed glmtools (typically, `glmval`)

# In[1]:


import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from glmtools.io.glm import GLMDataset
from glmtools.plot.grid import plot_glm_grid
import pdb 

# ## Use a sample data file included in glmtools

# In[10]:



from glmtools.test.common import get_sample_data_path

sample_path = get_sample_data_path()
samples = [
    "OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc",
]
samples = [os.path.join(sample_path, s) for s in samples]
samples = glob.glob('/glade/scratch/ahijevyc/noaa-goes16/GLM-L2-LCFA/2018/194/00/OR_GLM-L2*.nc')

filename = samples[0]


# ## Use data from the most recent minute or two
# 
# Requires siphon.
# 
# To load data via siphon from opendap, you must
# 
# `conda install -c conda-forge siphon`

# In[21]:


# Load data from the most recent minute or two!
if False:
    from siphon.catalog import TDSCatalog
    g16url = "http://thredds-test.unidata.ucar.edu/thredds/catalog/satellite/goes16/GRB16/GLM/LCFA/current/catalog.xml"
    satcat = TDSCatalog(g16url)
    filename = satcat.datasets[-1].access_urls['OPENDAP']


# ## Load the data

# In[12]:


glm = GLMDataset(filename)
print(glm.dataset)


# ## Flip through each flash, plotting each.
# 
# Event centroids are small black squares.
# 
# Group centroids are white circles, colored by group energy.
# 
# Flash centroids are red 'x's

# In[13]:


from glmtools.plot.locations import plot_flash

# # Find flashes in some location
# 
# There are hundreds of flashes to browse above, and they are randomly scattered across the full disk. Storms near Lubbock, TX at the time of sample data file had relatively low flash rates, so let's find those.

# In[17]:


flashes_subset = glm.subset_flashes(lon_range = (-100., -92), lat_range = (41, 45))
print(flashes_subset)

fl_id_vals = list(flashes_subset.flash_id.data)
fl_id_vals.sort()

fig, axes = plt.subplots()
pdb.set_trace()
tidx = 0
for flash_id in fl_id_vals:
    mapax,cb  = plot_glm_grid(fig, glm, tidx, ["flash_extent_density"])



