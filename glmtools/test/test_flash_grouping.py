import numpy as np

from numpy.testing import assert_equal

from .common import get_test_dataset
    
def test_flash_id_sanity():
    glm, flash_ids = get_test_dataset()
    
    # Make sure the flash IDs returned are the same as the flash IDs requested
    fls = glm.get_flashes(flash_ids)
    assert_equal(flash_ids, fls.flash_id.data)

    # Make sure the group parent flash IDs returned are the same as the flash IDs requested
    group_parents = np.unique(fls.group_parent_flash_id)
    assert_equal(flash_ids, group_parents)
    
    # Now that we know that all groups correspond to the flashes they should, make sure that we have the right events for those groups
    group_ids = fls.group_id.data.copy()
    group_ids.sort()
    
    event_parents = np.unique(fls.event_parent_group_id)
    event_parents.sort()
    assert_equal(group_ids, event_parents)

def test_flash_ids_for_events():
    glm, flash_ids = get_test_dataset()
    
    flash_ids_for_events = glm.dataset['event_parent_flash_id'].data
    
    n_events = glm.dataset.dims['number_of_events']
    assert_equal(flash_ids_for_events.shape[0], n_events)
    
    unq_fl_ids = np.unique(flash_ids_for_events)
    unq_fl_ids.sort()
    
    assert_equal(flash_ids, unq_fl_ids)