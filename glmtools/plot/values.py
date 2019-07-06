from matplotlib.colors import LogNorm, Normalize

display_params = {}

display_params['flash_centroid_density'] = {
    'product_label':"GOES-16 GLM Flash Centroid Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=10),
    'file_tag':'flash_centroid',
    'format_string':'{0:3.0f}'
}

display_params['flash_extent_density'] = {
    'product_label':"GOES-16 GLM Flash Extent Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=50),
    'file_tag':'flash_extent',
    'format_string':'{0:3.0f}'
}

display_params['group_centroid_density'] = {
    'product_label':"GOES-16 GLM Group Centroid Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=10),
    'file_tag':'group_centroid',
    'format_string':'{0:3.0f}'
}

display_params['group_extent_density'] = {
    'product_label':"GOES-16 GLM Group Extent Density (count)",
    'glm_norm':LogNorm(vmin=1, vmax=500),
    'file_tag':'group_extent',
    'format_string':'{0:3.0f}'
}
display_params['event_density']=display_params['group_extent_density']

display_params['total_energy'] = {
    'product_label':"GOES-16 GLM Total Energy (nJ)",
    'glm_norm':LogNorm(vmin=1e-8, vmax=1e-3),
    'file_tag':'total_energy',
    'format_string':'{0:3.1e}'
}

display_params[ 'average_flash_area'] = {
    'product_label':"GOES-16 GLM Average Flash Area (km$^2$)",
    'glm_norm':LogNorm(vmin=50, vmax=.5e4),
    'file_tag':'flash_area',
    'format_string':'{0:3.0f}'
}

display_params['average_group_area'] = {
    'product_label':"GOES-16 GLM Average Group Area (km$^2$)",
    'glm_norm':LogNorm(vmin=50, vmax=.5e4),
    'file_tag':'group_area',
    'format_string':'{0:3.0f}'
}