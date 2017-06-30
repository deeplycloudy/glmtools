import numpy as np
from numpy.testing import assert_equal

from .common import get_four_level_data

from glmtools.io.traversal import OneToManyTraversal

def test_replicate_parent_ids():
    d = get_four_level_data()
    
    entity_vars = ('storm_id', 'flash_id', 'stroke_id', 'trig_id')
    parent_vars = ('flash_parent_storm_id', 
                   'stroke_parent_flash_id', 
                   'trig_parent_stroke_id')
    traversal = OneToManyTraversal(d, entity_vars, parent_vars)
    
    trig_parent_storm_ids = traversal.replicate_parent_ids('storm_id', 
                                                    'trig_parent_stroke_id')
    trig_parent_flash_ids = traversal.replicate_parent_ids('flash_id', 
                                                    'trig_parent_stroke_id')
    trig_parent_stroke_ids = traversal.replicate_parent_ids('stroke_id', 
                                                    'trig_parent_stroke_id')

    print(d)
    assert_equal(d['trig_parent_storm_id'].data, trig_parent_storm_ids)
    assert_equal(d['trig_parent_flash_id'].data, trig_parent_flash_ids)
    assert_equal(d['trig_parent_stroke_id'].data, trig_parent_stroke_ids)