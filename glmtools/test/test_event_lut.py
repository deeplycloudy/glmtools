import numpy as np
from numpy.testing import assert_equal

from glmtools.io.glm import get_lutevents

from .common import get_test_dataset

def test_lut_events():
    glm, flash_ids = get_test_dataset()
    
    fls = check_flash_dataset(glm.dataset)
    assert_equal(fls.lutevent_id.shape[0], 24)

    sub = glm.subset_flashes()
    fls = check_flash_dataset(sub)
    assert_equal(fls.lutevent_id.shape[0], 24)

    some = glm.get_flashes(flash_ids[2:4])
    fls = check_flash_dataset(some)
    assert_equal(fls.lutevent_id.shape[0], 23)

    sub = glm.subset_flashes(lat_range=(-20, 0), lon_range=(-110, -90))
    fls = check_flash_dataset(sub)
    assert_equal(fls.lutevent_id.shape[0], 0)

    sub = glm.subset_flashes(lat_range=(35, 35.05), lon_range=(-95, -94.9))
    fls = check_flash_dataset(sub)
    assert_equal(fls.lutevent_id.shape[0], 24)


def check_flash_dataset(fls):
    # Direcly sum all event energy and get event count
    fls = get_lutevents(fls)
    print(fls)
    total_energy = fls.event_energy.data.sum()
    total_count = fls.event_id.shape[0]

    # Sum the energy from each lut entry
    total_energy_lut = 0
    total_count_lut = 0

    # Get the lut event IDs that go with each actual event
    lutevent_ids = np.unique(fls.event_parent_lutevent_id.data)

    # Event counts and total energy for each lut event should sum to the total
    # calculated from the indvidual events.
    for lut_id in lutevent_ids:
        lut_row = fls.loc[{'lutevent_id':[lut_id]}]
        assert_equal(lut_row.lutevent_id.shape[0], 1)
        total_energy_lut += lut_row.lutevent_energy.data
        total_count_lut += lut_row.lutevent_count.data
    
    assert_equal(total_energy, total_energy_lut)
    assert_equal(total_count, total_count_lut)
    
    return fls
        
    
    
    