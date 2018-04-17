"""

Depends on pyclipper: https://github.com/greginvm/pyclipper
which is a simple Python wrapper for Angus Johnson's Clipper.
http://www.angusj.com/delphi/clipper.php#code

Later: check to see why Clipper isn't used in Agg instead of Sutherland-Hodgman. 
There is an Agg demo in the Clipper source.
Would be good to do a demo with some radar data to see just how fast this could be
vs. pcolormesh.


Also, there all mesh quads could probably be processed simultaneously if we
made the change to clipper described below. 

https://stackoverflow.com/questions/46235176/clipperlib-clip-multiple-squares-with-rectangle-produces-1-result

"Paul, the latest version of Clipper (still in development but solid and faster
and unlikely to change much now before formal release) doesn't have any polygon
merging so should do what you want ...
sourceforge.net/p/polyclipping/code/HEAD/tree/sandbox/Clipper2.
Alternatively you could comment out the the JoinCommonEdges() statement in the
ExecuteInternal method in the old Clipper."
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from glmtools.io.ccd import create_pixel_lookup
import pyclipper
from pyclipper import Pyclipper, PolyTreeToPaths, ClosedPathsFromPolyTree, scale_to_clipper, scale_from_clipper

def vectorized_poly_area(x,y):
    """ Calculate the area of a non-self-intersecting planar polygon.
        x0y1 - x0y0 + x1y2 - x2y1 + x2y3 - x2y2 + ... + xny0 - x0yn
    
        Operate on all N polygons with M vertices, where x and y
        both have shape N, M
    """
    # determinant
    det = x[:, :-1]*y[:, 1:] - x[:, 1:]*y[:, :-1]
    area = det.sum(axis=1)
    # wrap-around terms in determinant
    area += x[:, -1]*y[:, 0] - x[:, 0]*y[:,-1]
    area *= 0.5
    return area
    
def poly_area(x,y):
    """ Calculate the area of a non-self-intersecting planar polygon.
        x0y1 - x0y0 + x1y2 - x2y1 + x2y3 - x2y2 + ... + xny0 - x0yn
    """
    det = x[:-1]*y[1:] - x[1:]*y[:-1] # determinant
    area = det.sum()
    area += x[-1]*y[0] - x[0]*y[-1] # wrap-around terms in determinant
    area *= 0.5
    return area

def lines_from_quadmesh(xedge, yedge, scale=True):
    """ Given the edges of a quadmesh, return (h_lines, v_lines) corresponding
    to the edges along each dimension.
    
    If scale is True (default), convert coordinates to scaled integers
    as appropriate for clipper.
    """

    # Get the lines along each dimension. h, v need not
    # correspond to the horizontal and vertical dimensions.
    # It's shorter than writing coord0, coord1
    lines = np.dstack((xedge, yedge))
    nh, nv, nd = lines.shape
    h_lines = [lines[i,:,:] for i in range(nh)]
    v_lines = [lines[:,j,:] for j in range(nv)]
    
    if scale:
        # The scale factor could be adjusted, but by default it results in 
        # a range of +/- 2 ** 31 with a precision of 2 ** -31. See the docs.
        h_lines = scale_to_clipper(h_lines)
        v_lines = scale_to_clipper(v_lines)
    return h_lines, v_lines


            
def polys_from_quadmesh(x, y):
    """ x, y are M x N arrays giving the edge locations for
        a quadrilateral mesh of Nq = (M-1) x (N-1) implied quad faces.
        
        returns
        vertices : ndarray, shape (Nv---, 3) - Vertex coordinates. 
        faces : ndarray, shape (Nf---, 3) - Indices into the vertex array.

    """
    M, N = x.shape
    Nv = M * N
    Nq = (M-1) * (N-1)
    
    v = np.dstack((x, y)) # M, N, 2
    
    quads  = np.empty((M-1, N-1, 4, 2), dtype='f8')
    
    quads[:,:,0,:] = v[:-1, :-1, :]
    quads[:,:,1,:] = v[:-1,  1:, :]
    quads[:,:,2,:] = v[ 1:,  1:, :]
    quads[:,:,3,:] = v[ 1:, :-1, :]
    return quads

class QuadMeshSubset(object):
    """ Given a quadmesh, create a KDTree for use in grabbing a subset of the
    quadrilaterals comprising the mesh. We don't assume a regular grid,
    i.e., the target polygons might be non-rectilinear as well.
    
    Uses the centroid of the quads because we only need to get close enough.
    """
    def __init__(self, xedge, yedge, n_neighbors=12, X_ctr=None, Y_ctr=None,
        regular=False):
        """ xedge and yedge are the x and y edge coordinates of a quadrilateral mesh

        min_neighbors is the minimum number of neighbors returned
        
        
        """
        # Next use a KDTree to find the subset of quads nearest the target polygon. We want to get the k quads nearest a point, with k determined based on the typical (max) area of a polygon P and the typical (min) area of the  target grids G. 
    
        # We will assume that no observed polygons are narrower than the square root of the target grid area. Worst case would be a polygon that fit entirely with one row or column. Then the number of polygons to find would be min((sqrt(P)/sqrt(G)+1)^2, 4). 4 is minimum in case of the tiniest polygon across an arbitrarily large grid.
    
        # By example, for a 1 km grid and an 8 km poly, the max height would be 8, pad with one, to get 81 grid cells. For a 2 km grid and an 8 km poly, 4+1 grid cells or 25 grid cells. For an 8 km grid, 4 grid cells.
        
        self.regular = regular
        self.n_neighbors = n_neighbors
        self.xedge = xedge
        self.yedge = yedge
        if X_ctr is None:
            X_ctr = (xedge[:-1, :-1] + xedge[1:, 1:] +
                     xedge[1:, :-1] + xedge[:-1, 1:])/4.0
        if Y_ctr is None:
            Y_ctr = (yedge[:-1, :-1] + yedge[1:, 1:] +
                     yedge[1:, :-1] + yedge[:-1, 1:])/4.0
        self.X_ctr = X_ctr
        self.Y_ctr = Y_ctr
            
        print('Calculating polygons from mesh ...')
        # all_quad_idx0, all_quad_idx1 = (np.arange(X_ctr.shape[0]),
        #                                            np.arange(X_ctr.shape[1]))
        # quads = [q for q in self.gen_polys(
        #             all_quad_idx0.flatten(), all_quad_idx1.flatten())]
        # self.quads = np.asarray(quads) # shape = (N0 x N1, 4, 2)
        # quad_x = self.quads[:,:,0]
        # quad_y = self.quads[:,:,1]
        # self.quads.shape = self.X_ctr.shape + (4,2)
        
        self.quads = polys_from_quadmesh(xedge, yedge)
        nq = self.quads.shape[0] * self.quads.shape[1]
        print('    ... {0} quads in mesh ...'.format(nq))
        self.quads_flat = self.quads.view()
        self.quads_flat.shape = (nq, 4, 2) # flatten the first two dimensions
        self.quad_areas = vectorized_poly_area(self.quads_flat[:, :, 0],
                                               self.quads_flat[:, :, 1])
        # self.quad_areas = np.abs(np.fromiter(
            # (poly_area(q[:,0], q[:,1]) for q in self.quads_flat),
            # dtype='f8', count=nq))
        self.quad_areas.shape = self.X_ctr.shape
        
        # if min_mesh_size is None:
#             min_mesh_size = self.quad_areas.min()
#         print("Min mesh size is", min_mesh_size, " and max poly area is ", max_poly_area)
#         P, G = max_poly_area, min_mesh_size
## self.min_neighbors = max(int((np.sqrt(P)/np.sqrt(G)+1)**2), 8)
        if regular:
            print('    ... determining regular grid arrangement ...')
            # if we have a regular grid, by definition one of the
            # two dimensions has constant values. Depending on how meshgrid
            # was used, that may be along either dimension one or two
            if np.allclose(xedge[:,0].mean() - xedge[:,0], 0.0):
                # constant values indicate this
                # is not the increasing dimension for x
                self.X_ctr1d = self.X_ctr[0,:]
                self.Y_ctr1d = self.Y_ctr[:,0]
                self.X_edge1d = self.xedge[0,:]
                self.Y_edge1d = self.yedge[:,0]
                self.X_increasing_dim = 1
                self.Y_increasing_dim = 0
            else:
                # this is the increasing dimension
                self.X_ctr1d = self.X_ctr[:,0]
                self.Y_ctr1d = self.Y_ctr[0,:]
                self.X_edge1d = self.xedge[:,0]
                self.Y_edge1d = self.yedge[0,:]
                self.X_increasing_dim = 0
                self.Y_increasing_dim = 1
            self.Xi1d = np.arange(0, len(self.X_ctr1d))
            self.Yi1d = np.arange(0, len(self.Y_ctr1d))
            self.N_X_ctr = self.X_ctr1d.shape[0]
            self.N_Y_ctr = self.Y_ctr1d.shape[0]
            print('    ... done.')
        else:
            # We reuse this function that was spec'd in terms of lon/lat, but it's
            # actually general for a quadmesh in any coordinate system.
            print('    ... constructing search tree ... be patient ...')
            self.tree, self.Xi, self.Yi = create_pixel_lookup(X_ctr, Y_ctr, leaf_size=16)
            print('    ... done.')
        
        
    # def gen_polys(self, xidx, yidx):
    #     lines = self.xyedge
    #
    #     for i in xidx:
    #         for j in yidx:
    #             # The code cries out for vectorization
    #             quad = lines[i,j,:], lines[i,j+1,:], lines[i+1,j+1,:], lines[i+1,j,:]
    #             yield quad
        
    def query_tree(self, x):
        """ x is a 2-tuple or two element array in the same coordinates as self.xedge, self.yedge
        
        returns dist, quad_x_idxs, quad_y_idxs. These are indices into the pixel center arrays
        but also work as the low-index corner of the edge arrays
        """
        if self.regular:
            # This is a single point query, so the xmin, xmax are the same.
            # Same goes for the y coordinate.
            bbox = [x[0],x[0],x[1],x[1]]
            dist = None
            quad_x_idx, quad_y_idx = self.quads_in_bbox_fast(bbox)
        else:
            # idx = self.tree.query_radius([x], r=self.n_neighbors)
            dist, idx = self.tree.query([x], k=self.n_neighbors)
            quad_x_idx, quad_y_idx = self.Xi[idx], self.Yi[idx]
            # print('idx, quad_x_idx, quad_y_idx', idx, quad_x_idx, quad_y_idx)
        return dist, quad_x_idx, quad_y_idx
        
    def quads_nearest(self, x):
        """ Find quads from the mesh corresponding to the neighbors nearest x.
            x is a 2-tuple or two element array in the same coordinates as 
            self.xedge, self.yedge

            returns quads, quad_x_idxs, quad_y_idxs. The indices are into the
            pixel center arrays self.X_ctr, self.Y_ctr
            but also work as the low-index corner of the edge arrays
        """
        dists, quad_x_idx, quad_y_idx = self.query_tree(x)
        # quads = [q for q in self.gen_polys(quad_x_idx, quad_y_idx)]
        # Squeeze a leading dimension of 1 on x/y_idx and therefore on quads
        quads = self.quads[quad_x_idx, quad_y_idx, :, :].squeeze()
        return quads, quad_x_idx.squeeze(), quad_y_idx.squeeze()
    
    def quads_in_bbox_fast(self, bbox, pad=2):
        xlim = bbox[:2]
        ylim = bbox[2:]
        # print('bbox is', bbox)
        
        # We have xmin, xmax, ymin, ymax and want to index the quads array
        # with the indices that are between the minimum and maximum
        # goodx = ((xlim[0] <= self.X_edge1d[:-1]) & (xlim[1] >= self.X_edge1d[1:]))
        # goody = ((ylim[0] <= self.Y_edge1d[:-1]) & (ylim[1] >= self.Y_edge1d[1:]))
        # idxx, = np.where(goodx)
        # idxy, = np.where(goody)
        # When given an edge array, (digitize - 1) will return < 0 for 
        # values less than the leftmost edge and (n_bins = n_edges-1)
        # aedge = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # actr = array([ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        # idxx = np.digitize((-1, 0, 0.5, 2.5, 9, 10), aedge, right=True) - 1
        # valid = (idxx>=0) & (idxx < actr.shape[0])
        # print(valid) -> [False, False,  True,  True,  True, False]
        idxx = np.digitize(xlim, self.X_edge1d, right=True) - 1
        idxy = np.digitize(ylim, self.Y_edge1d, right=True) - 1
        validx = (idxx >= 0) & (idxx < self.N_X_ctr)
        validy = (idxy >= 0) & (idxy < self.N_Y_ctr)
        idxx = idxx[validx]
        idxy = idxy[validy]
        # print('indices x', idxx, self.X_edge1d.min(), self.X_edge1d.max())
        # print('indices y', idxy, self.Y_edge1d.min(), self.Y_edge1d.max())
        if (len(idxx) > 0) & (len(idxy) > 0):
            xi_min = max(idxx.min() - pad, 0)
            xi_max = min(idxx.max() + 1 + pad, self.N_X_ctr)
            yi_min = max(idxy.min() - pad, 0)
            yi_max = min(idxy.max() + 1 + pad, self.N_Y_ctr)
            sl_quad = [0, 0]
            sl_quad_xi = slice(xi_min, xi_max)
            sl_quad_yi = slice(yi_min, yi_max)
            sl_quad[self.X_increasing_dim] = self.Xi1d[sl_quad_xi]
            sl_quad[self.Y_increasing_dim] = self.Yi1d[sl_quad_yi]
        else:
            sl_quad = [self.Xi1d[0:0], self.Yi1d[0:0]]
        q_shp = np.ones((sl_quad[0].shape[0], sl_quad[1].shape[0]), dtype=int)        
        vals = ((sl_quad[0][:,None]*q_shp).flatten(), 
                (sl_quad[1][None,:]*q_shp).flatten())
        return vals

    def quads_in_bbox(self, bbox):
        """ bbox is xmin, xmax, ymin, ymax """
        if self.regular:
            # quad_x_idx, y_idx refer to the first and second dimensions
            # of quads, and may not match X_ctr1d, Y_ctr1d.
            quad_x_idx, quad_y_idx = self.quads_in_bbox_fast(bbox)
            quads = self.quads[quad_x_idx, quad_y_idx, :, :]
            return quads, quad_x_idx, quad_y_idx
        
        xlim = bbox[:2]
        ylim = bbox[2:]
        # These will miss quads that straddle the edge of the bbox if their
        # centers are outside of the bbox
        # goodx = (self.X_ctr >= xlim[0]) & (self.X_ctr <= xlim[1])
        # goody = (self.Y_ctr >= ylim[0]) & (self.Y_ctr <= ylim[1])
        goodx = np.zeros(self.quads.shape[0:2], dtype=bool)
        goody = np.zeros(self.quads.shape[0:2], dtype=bool)

        for qi in range(4):
            # loop over the quad corners, keeping quads with at least one corner in bounds
            goodx |= ((self.quads[:,:,qi,0] >= xlim[0]) &
                      (self.quads[:,:,qi,0] <= xlim[1])) 
            goody |= ((self.quads[:,:,qi,1] >= ylim[0]) &
                      (self.quads[:,:,qi,1] <= ylim[1]))
        quad_x_idx, quad_y_idx = np.where((goodx & goody))
        quads = self.quads[quad_x_idx, quad_y_idx, :, :]
        return quads, quad_x_idx, quad_y_idx

def clip_polys_by_one_poly(polys, p, scale=True):
    """ polys: a list of polygons
        p: the polygon with which to clip polys
        scale: convert floating point polygon coordinates to ints required
            by the underlying clipper library.

        Returns results, sub_poly_count
        results: all polygons that were part of p but were chopped by the
            other polys

        sub_poly_count: how many sub-polygons were found for each
            poly in polys. Can be used with np.repeat to replicate a
            list of values associated with each original poly in polys.
    """
    pc = pyclipper.Pyclipper()

    if scale:
        polys = scale_to_clipper(polys)
        p = scale_to_clipper(p)

    # Each individual sub-poly is stored here.
    results=[]
    # For any polygon p, zero, one or more than one polygon may be returned
    # for each poly in polys (depending on overlap and the complexity of p).
    # Keep track of how many sub-polygons were found for each poly in polys
    sub_polys_per_poly=[]
    for q in polys:
        pc.AddPath(q, pyclipper.PT_SUBJECT, True)
        pc.AddPath(p, pyclipper.PT_CLIP, True)
        clip_polys = pc.Execute(clip_type=pyclipper.CT_INTERSECTION)
        if scale:
            clip_polys = scale_from_clipper(clip_polys)
        sub_polys_per_poly.append(len(clip_polys))
        results.extend(list(clip_polys))
        pc.Clear()
    return results, sub_polys_per_poly

def join_polys(polys, scale=True):
    """ Given a list of polygons, merge them (union) and return a list
        of merged polygons
    """
    pc = pyclipper.Pyclipper()

    if scale:
        polys = scale_to_clipper(polys)

    results=[]
    pc.AddPaths(polys, pyclipper.PT_SUBJECT, True)
    clip_polys = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, 
        pyclipper.PFT_NONZERO)
    if scale:
        clip_polys = scale_from_clipper(clip_polys)
    results.extend([cp for cp in clip_polys]) 
    pc.Clear()
    return results

class QuadMeshPolySlicer(object):
    def __init__(self, mesh):
        """ mesh is an instance of QuadMeshSubset
        
        e.g., QuadMeshSubset(xedge, yedge, n_neighbors=12)
        """
        self.mesh = mesh

    def slice(self, polys, bbox=None):
        """ polys is an (N, M, 2) array or N-element list of (M,2) arrays
        of N polygons with M vertices in two dimensions.
        
        if bbox = (xmin, xmax, ymin, ymax) is present, slice with the mesh
        subset within that bbox. If bbox is true, use the dimensions of each
        poly to figure out the bbox. Otherwise, n nearest neighbors as
        configured for the QuadMeshSubset object are returned.
        
        Returns (sliced_poly_list, orig_poly_areas)
        
        orig_poly_areas are the areas of the polygons passed in. These are
        calculated as a byproduct.
        
        sliced_poly_list is a list of length N for each of the original N polys.
        Each list entry contains a tuple giving the properties of each sliced
        quad that intersects the original poly. The list entries are
        (sub_quad_polys, frac_areas, (x_idxs, y_idxs))
        Each element of the tuple above is of length Q for the Q quads 
        intersected by the original poly.
        
        Each list entry in sub_quad_polys is another list of vertices for each 
        of the Q quads having the fractional areas given in frac_areas.
        
        The indices index mesh.X_ctr, mesh.Y_ctr but also work as the
        low-index corner of the mesh edge arrays. Therefore, the indices can 
        be used to retrieve values from data arrays having the geometry of the
        original mesh.
        
        Example
        -------
        # Create the meshgrid that will chop up the polygons below
        x = np.arange(50)
        y = np.arange(60)+10
        X,Y=np.meshgrid(x,y)

        # Assign some values to each grid cell
        vals = np.random.rand(X.shape[0]-1, X.shape[1]-1)
        print(X.shape, Y.shape, vals.shape)

        # Set up the meshlookup and prepare to slice.
        mesh = QuadMeshSubset(X, Y, n_neighbors=20)
        slicer = QuadMeshPolySlicer(mesh)

        a_poly = [(.5,12), (1.7,13), (1.9,12.5), (.9, 11)]
        # Create a bunch of random polygons
        N_polys = 10
        polys = np.asarray([a_poly] * N_polys)
        polys += .8*x.shape[0]*np.random.rand(N_polys)[:,None,None]
        chopped_polys, orig_poly_areas = slicer.slice(polys)

        # Stack together all subpolys from each of the 10 original polys
        def gen_polys(chopped_polys):
            for subquads, areas, (x_idxs, y_idxs)  in chopped_polys:
                for subquad, area, x_idx, y_idx in zip(subquads, areas, x_idxs, y_idxs):
                    print('subquad', subquad)
                    print('area', area)
                    print('idx', x_idx, y_idx)
                    yield (subquad, area, (x_idx, y_idx))
        
        good_polys = [p for p in gen_polys(chopped_polys)]
        
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        fig, axs = plt.subplots(1,2, figsize=(16,10), sharex=True, sharey=True)
        ax, ax1 = axs[0], axs[1]
        vmin, vmax = 0, 1

        # Chopped polys
        pm = ax.pcolormesh(X, Y, vals*0, edgecolor='black', alpha=0.1, cmap='cubehelix_r', vmin=vmin, vmax=vmax)
        ax.plot(mesh.X_ctr, mesh.Y_ctr, '.k')
        patches = [Polygon(p, True) for p, area, ctrs in good_polys]
        patch_coll = PatchCollection(patches, edgecolors='red', norm=pm.norm, cmap=pm.cmap, alpha=1.0)
        patch_vals = np.asarray([vals[ctrs[0], ctrs[1]] for p, area, ctrs in good_polys])
        print(patch_vals)
        patch_coll.set_array(patch_vals)
        ax.add_collection(patch_coll)

        # Original polys
        pm1 = ax1.pcolormesh(X,Y, vals, edgecolor='none', alpha=1.0, norm=pm.norm, cmap=pm.cmap)
        patches = [Polygon(p, True) for p in polys]
        patch_coll = PatchCollection(patches, edgecolors='red', facecolors='none', norm=pm.norm, cmap=pm.cmap, alpha=1.0)
        patch_coll.set_array(np.fromiter((0 for p in polys), dtype=float))
        ax1.add_collection(patch_coll)
        """
        if bbox == True:
            recalc_bbox = True
            # else use the bbox that is specified exactly
        
        poly_arr = [np.asarray(p) for p in polys]
        areas = [np.abs(poly_area(p[:,0], p[:,1])) for p in poly_arr]
        poly_ctr = [p.mean(axis=0) for p in poly_arr]
    
        all_quads = []
        for poly, pctr in zip(poly_arr, poly_ctr):
            if bbox is not None:
                if recalc_bbox:
                    mins = np.min(poly, axis=0)
                    maxs = np.max(poly, axis=0)
                    bbox = mins[0], maxs[0], mins[1], maxs[1]
                    # print('using bbox', bbox)
                quads, quad_x_idx, quad_y_idx = self.mesh.quads_in_bbox(bbox)
            else:
                quads, quad_x_idx, quad_y_idx = self.mesh.quads_nearest(pctr)
            # each of the return values above has the same shape in the first two dimensions. They are the quad indices that go with the original mesh.
            # print('quad shapes', quads.shape, quad_x_idx.shape, quad_y_idx.shape)
            nq = quads.shape[0] #* quads.shape[1]

            quad_x_idx.shape = (nq,)
            quad_y_idx.shape = (nq,)
            quads.shape = (nq, 4, 2) # so that we can do "for q in quads"
            all_quads.append((quads, quad_x_idx, quad_y_idx))
            
        sub_poly_args = poly_arr, areas, all_quads
        sub_poly_args = tuple(zip(poly_arr, areas, all_quads))

        # sub_polys = list(map(make_sub_polys, sub_poly_args))
        # pool = ProcessPoolExecutor(max_workers=6)
        # with pool:
            # sub_polys = list(pool.map(make_sub_polys, sub_poly_args, chunksize=100))

        sub_polys = list(run_pool_map(make_sub_polys, sub_poly_args))
        return sub_polys, areas

    def quad_frac_from_poly_frac_area(self, frac_areas, total_area, 
            quad_x_idx, quad_y_idx):
        """ Slicing a polygon with the quadmesh returns one or more sub-quads
        and their frac_areas as a fraction of the polygon's total_area. The
        sub-quads came from the original mesh at quad_x_idx, quad_y_idx.
        
        Calculate the fraction of each original quad covered by the sub-quad.
        """
        sub_quad_areas = np.asarray(frac_areas)*total_area
        quad_areas = self.mesh.quad_areas[quad_x_idx, quad_y_idx]
        quad_frac = sub_quad_areas/quad_areas
        return quad_frac



def make_sub_polys(args):
    poly, area, q_dat = args
    quads, quad_x_idx, quad_y_idx = q_dat
    all_clip_polys, count_per_quad = clip_polys_by_one_poly(quads, poly)
    clip_x_idx = np.repeat(quad_x_idx, count_per_quad)
    clip_y_idx = np.repeat(quad_y_idx, count_per_quad)
    clip_polys = [np.asarray(cp, dtype='f8').squeeze() 
                  for cp in all_clip_polys]
    frac_areas = [np.abs(poly_area(p[:,0], p[:,1])/area) for p in clip_polys]
    total_fraction = np.asarray(frac_areas).sum()*100
    if (np.abs(total_fraction-100) > 0.1):
        print(not_enough_neighbors_err.format(total_fraction))
    return (clip_polys, frac_areas, (clip_x_idx, clip_y_idx))

not_enough_neighbors_err = """Polygon only {0} percent covered by quads"""
        


def run_pool_map(f,a):
    pool = ProcessPoolExecutor(max_workers=4)
    return pool.map(f,a,chunksize=100)
# def dummy_work(a):
#     return a
# value = run_pool_map(dummy_work, list(range(10)))
