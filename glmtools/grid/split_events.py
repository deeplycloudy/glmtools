import numpy as np
import xarray as xr

def gen_split_events(chopped_polys, poly_areas, slicer, event_ids=None):
    """
    chopped_polys is a list of N polygons whose elements contain the sub-polys of each polygon.
        It is the data structure created by QuadMeshPolySlicer.slice
    event_ids are N corresponding event_ids
    """
    if event_ids is None: event_ids = range(len(chopped_polys))
    
    for (subquads, frac_areas, (x_idxs, y_idxs)), total_area, evid in zip(chopped_polys, poly_areas, event_ids):
        quad_fracs = slicer.quad_frac_from_poly_frac_area(
                        frac_areas, total_area, x_idxs, y_idxs)

        for subquad, frac_area, x_idx, y_idx, quad_area in zip(subquads, frac_areas, x_idxs, y_idxs, quad_fracs):
#             print('-------')
#             print('subquad', subquad)
#             print('frac_area, quad_frac_area', frac_area, quad_area)
#             print('evid, idx', evid, x_idx, y_idx)
            yield (subquad, frac_area, quad_area, (x_idx, y_idx), evid)

def split_event_data(split_polys, poly_areas, slicer, event_ids):
    """
    split_polys is a list of N polygons whose elements contain the sub-polys of each polygon.
        It is the data structure created by QuadMeshPolySlicer.slice
    event_ids are N corresponding event_ids
    """

    # fromiter would run faster if we could precalculate the count, though
    # doing so would require iterating to sum some sizes, so it's not clear
    # if there's much net benefit.
    dtype = np.dtype([
#         ('poly','f8',(4,2)), # polys could be any number of verts, so don't use.
        ('poly_ctr', 'f8', (2,)),
        ('event_frac_area', 'f8'),
        ('mesh_frac_area', 'f8'),
        ('mesh_idx', 'i8', (2,)),
#         ('mesh_y_idx', 'i8'),
        ('event_id', 'u8')
    ])
    
    parts_of_split_polys = [p for p in 
        gen_split_events(split_polys, poly_areas, slicer, event_ids=event_ids)]
    
    # Each element here will be an (n_verts, 2) array of polygon vertex locations.
    sub_polys = [np.asarray(sp[0],dtype='f8') for sp in parts_of_split_polys]

    # These are frac_area, quad_area, (x_idx, y_idx), evid - i.e., 
    # the parts with the same length that can be turned into an array.
    split_event_property_iter = (sp[1:] for sp in parts_of_split_polys)
    
    n_sub_polys = len(sub_polys)
    
#     for sp, (frac_area, quad_area, idxs, evid) in zip(sub_polys, split_event_property_iter):
#         sp.mean(axis=0)
    
    split_event_iter = ((sp.mean(axis=0), frac_area, quad_area, idxs, evid)
                        for (sp, (frac_area, quad_area, idxs, evid)) in 
                        zip(sub_polys, split_event_property_iter))
    
    d = np.fromiter(split_event_iter, dtype=dtype, count=n_sub_polys)

    return sub_polys, d


def split_event_dataset_from_props(props):
    """ props is the numpy array with named dtype returned by split_event_dataset """
    
    dims = ('number_of_split_events',)
    d ={
        'split_event_lon': {'dims':dims, 'data':props['poly_ctr'][:,0]},
        'split_event_lat': {'dims':dims, 'data':props['poly_ctr'][:,1]},
        'split_event_mesh_area_fraction': {'dims':dims, 'data':props['mesh_frac_area']},
        'split_event_area_fraction': {'dims':dims, 'data':props['event_frac_area']},
        'split_event_mesh_x_idx': {'dims':dims, 'data':props['mesh_idx'][:,0]},
        'split_event_mesh_y_idx': {'dims':dims, 'data':props['mesh_idx'][:,1]},
        'split_event_parent_event_id': {'dims':dims, 'data':props['event_id']},
    }
    return xr.Dataset.from_dict(d)


def replicate_and_weight_split_event_dataset(glm, split_event_dataset,
                                             names=['event_energy', 'event_time_offset',
                                                    'event_parent_flash_id', 'event_parent_group_id'],
                                             weights={'event_energy':'split_event_area_fraction'}
                                            ):
    
    replicated_event_ids = split_event_dataset.split_event_parent_event_id

    # replicate the event radiances using the replicated event_ids. 
    # This chunk is stolen from the traversal.replicate_parent_id class
    # and should be moved back there after it's generalized 
    grouper = glm.entity_groups['event_id']
    e_idx = [grouper.groups[eid] for eid in replicated_event_ids.data]
    e_idx_flat = np.asarray(e_idx).flatten()
    
    for name in names:
        new_name = 'split_' + name
        new_var = getattr(glm.dataset, name)[e_idx_flat]
        if name in weights:
            weight_var = getattr(split_event_dataset, weights[name])
            # dimension names won't match, but lengths should.
            new_var *= weight_var.data
        # add variable to the dataset
        split_event_dataset[new_name] = new_var 
        
    return split_event_dataset
