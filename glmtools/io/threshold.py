import numpy as np
import xarray as xr
from glmtools.io.glm import GLMDataset, discretize_2d_location


def recalculate_group_data(glm_prune, glm):

    # Convert to dataframe since xarray is slow in sum over groupby.
    event_vars = ['event_energy', 'event_parent_group_id']
    glm_prune_event_df = glm_prune.dataset[event_vars].to_dataframe()

    idx_getter = ((glm_prune.entity_groups['group_id'].groups[gid][0],
                   glm.entity_groups['group_id'].groups[gid][0],
                   gid
                  )
                  for gid in glm_prune.entity_groups['group_id'].groups.keys())
    prune_group_idx, pre_group_idx, gids = map(np.asarray, zip(*idx_getter))

    prune_ev_sum = glm_prune_event_df.groupby('event_parent_group_id').sum()
    new_group_energy = np.asarray(prune_ev_sum['event_energy'][gids])
    pre_count = glm.dataset.group_child_event_count[pre_group_idx]
    prune_count = glm_prune.dataset.group_child_event_count[prune_group_idx]
    new_group_area = glm.dataset.group_area[pre_group_idx] * (prune_count/
                                                              pre_count)

    glm_prune.dataset.group_area[prune_group_idx] = new_group_area
    glm_prune.dataset.group_energy[prune_group_idx] = new_group_energy

    return glm_prune


def unique_event_loc_ids(ds, scale_factor=28e-6):
    event_x, event_y = ds.event_x.data, ds.event_y.data
    x_range, y_range = (-0.31, 0.31), (-0.31, 0.31)
    xy_id = discretize_2d_location(event_x, event_y,
                                   scale_factor, x_range, y_range)
    return xr.DataArray(xy_id, dims=['number_of_events',])


def recalculate_flash_data(glm_prune, glm):
    pre_unique_event_ids = unique_event_loc_ids(glm.dataset)
    glm.dataset['event_parent_loc_id'] = pre_unique_event_ids
    prune_unique_event_ids = unique_event_loc_ids(glm_prune.dataset)
    glm_prune.dataset['event_parent_loc_id'] = prune_unique_event_ids

    idx_getter = ((glm_prune.entity_groups['flash_id'].groups[fid][0],
                   glm.entity_groups['flash_id'].groups[fid][0],
                   fid)
                  for fid in glm_prune.entity_groups['flash_id'].groups.keys())
    prune_flash_idx, pre_flash_idx, fids = map(np.asarray, zip(*idx_getter))

    group_vars = ['group_parent_flash_id', 'group_energy']
    prune_group_df = glm_prune.dataset[group_vars].to_dataframe()
    prune_gr_sum = prune_group_df.groupby('group_parent_flash_id').sum()
    new_flash_energy = np.asarray(prune_gr_sum['group_energy'][fids])

    event_vars = ['event_parent_flash_id','event_parent_loc_id']
    prune_event_df = glm_prune.dataset[event_vars].to_dataframe()
    pre_event_df = glm.dataset[event_vars].to_dataframe()

    prune_event_unq = prune_event_df.groupby('event_parent_flash_id').nunique()
    prune_count = np.asarray(prune_event_unq['event_parent_loc_id'][fids])

    pre_event_unq = pre_event_df.groupby('event_parent_flash_id').nunique()
    pre_count = np.asarray(pre_event_unq['event_parent_loc_id'][fids])

    new_flash_area = glm.dataset.flash_area[pre_flash_idx] * (prune_count/
                                                              pre_count)

    glm_prune.dataset.flash_area[prune_flash_idx] = new_flash_area
    glm_prune.dataset.flash_energy[prune_flash_idx] = new_flash_energy

    return glm_prune


def filter_to_energy(glm, thresh, units='nJ'):
    """ Filter GLMDataset object to threshold thresh in units as given."""
    assert glm.dataset.event_energy.attrs['units'] == units
    above_thresh = glm.dataset.event_energy >= thresh
    # After selecting the events above threshold the remaining
    # event_parent_flash_ids are the flashes with at least one event above
    # threshold, and similarly for groups.
    # remaining_flashes =
    #      np.unique(glm.dataset.event_parent_flash_id[above_thresh])
    remaining_groups = np.unique(glm.dataset.event_parent_group_id[above_thresh])

    # By pruning to the groups, any flashes with no groups
    # are automatically dropped
    ds = glm.reduce_to_entities('group_id', remaining_groups)

    # After trimming the tree, the below-thresh events for any non-empty groups
    # remain in the tree, so remove them.
    above_thresh = ds.event_energy >= thresh
    ds = ds[{'number_of_events': above_thresh}]

    # Reinitialize the GLM dataset, recalculating the parent child data,
    # including the group counts. Don't modify the event energies or times from
    # what was already there.
    glm_prune = GLMDataset(ds, calculate_parent_child=True,
                           ellipse_rev=glm.ellipse_rev,
                           check_area_units=False, change_energy_units=False,
                           fix_bad_DO07_times=False)

    # Also need to recalculate the flash and group energies.
    # Areas will be wrong, and need to be recalculated in proportion to the
    # number of events removed.
    glm_prune = recalculate_group_data(glm_prune, glm)
    glm_prune = recalculate_flash_data(glm_prune, glm)

    return glm_prune