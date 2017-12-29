"""

Depends on pyclipper: https://github.com/greginvm/pyclipper
which is a simple Python wrapper for Angus Johnson's Clipper.
http://www.angusj.com/delphi/clipper.php#code

Later: check to see why Clipper isn't used in Agg instead of Sutherland-Hodgman. 
There is an Agg demo in the Clipper source.
Would be good to do a demo with some radar data to see just how fast this could be
vs. pcolormesh.
"""
import numpy as np
from glmtools.io.ccd import create_pixel_lookup
import pyclipper
from pyclipper import Pyclipper, PolyTreeToPaths, ClosedPathsFromPolyTree, scale_to_clipper, scale_from_clipper


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
    def __init__(self, xedge, yedge, n_neighbors=12, X_ctr=None, Y_ctr=None):
        """ xedge and yedge are the x and y edge coordinates of a quadrilateral mesh

        min_neighbors is the minimum number of neighbors returned
        
        
        """
        # Next use a KDTree to find the subset of quads nearest the target polygon. We want to get the k quads nearest a point, with k determined based on the typical (max) area of a polygon P and the typical (min) area of the  target grids G. 
    
        # We will assume that no observed polygons are narrower than the square root of the target grid area. Worst case would be a polygon that fit entirely with one row or column. Then the number of polygons to find would be min((sqrt(P)/sqrt(G)+1)^2, 4). 4 is minimum in case of the tiniest polygon across an arbitrarily large grid.
    
        # By example, for a 1 km grid and an 8 km poly, the max height would be 8, pad with one, to get 81 grid cells. For a 2 km grid and an 8 km poly, 4+1 grid cells or 25 grid cells. For an 8 km grid, 4 grid cells.
    
        self.n_neighbors = n_neighbors
    
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
        # self.quad_areas = np.asarray([np.abs(poly_area(q[:,0], q[:,1])) for q in self.quads])
        # self.quad_areas.shape = self.X_ctr.shape
        
        # if min_mesh_size is None:
#             min_mesh_size = self.quad_areas.min()
#         print("Min mesh size is", min_mesh_size, " and max poly area is ", max_poly_area)
#         P, G = max_poly_area, min_mesh_size
## self.min_neighbors = max(int((np.sqrt(P)/np.sqrt(G)+1)**2), 8)
        
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
        
        returns dist, quad_x_idx, quad_y_idx. These are indices into the pixel center arrays
        but also work as the low-index corner of the edge arrays
        """
        # idx = self.tree.query_radius([x], r=self.n_neighbors)
        dist, idx = self.tree.query([x], k=self.n_neighbors)
        quad_x_idx, quad_y_idx = self.Xi[idx], self.Yi[idx]
        print('idx, quad_x_idx, quad_y_idx', idx, quad_x_idx, quad_y_idx)
        return dist, quad_x_idx, quad_y_idx
        
    def quads_nearest(self, x):
        """ return a list of quads from the mesh corresponding to the neighbors nearest x.
            x is a 2-tuple or two element array in the same coordinates as self.xedge, self.yedge
        """
        dists, quad_x_idx, quad_y_idx = self.query_tree(x)
        # quads = [q for q in self.gen_polys(quad_x_idx, quad_y_idx)]
        quads = self.quads[quad_x_idx, quad_y_idx, :, :]
        print('Nearest quads')
        for q in quads:
            print(q)
        return quads, quad_x_idx, quad_y_idx

def clip_polys_by_one_poly(polys, p, scale=True):
    pc = pyclipper.Pyclipper()
    open_path = False
    closed_path = True

    if scale:
        polys = scale_to_clipper(polys)
        p = scale_to_clipper(p)

    results = []
    for q in polys:
        pc.AddPath(q, pyclipper.PT_SUBJECT, closed_path)
        pc.AddPath(p, pyclipper.PT_CLIP, closed_path)
        clip_polys = pc.Execute(clip_type=pyclipper.CT_INTERSECTION)
        if scale:
            clip_polys = scale_from_clipper(clip_polys)
        results.append(clip_polys)    
        pc.Clear()
    return results

class QuadMeshPolySlicer(object):
    def __init__(self, mesh):
        """ mesh is an instance of QuadMeshSubset
        
        e.g., QuadMeshSubset(xedge, yedge, n_neighbors=12)
        """
        self.mesh = mesh
        
    def slice(self, polys):
        """ polys is an (N, M, 2) array of N polygons with M vertices in two dimensions
        
            Returns a list of length N whose elements are (polys, areas) for each quad
            that intersects the original N polys
        """
        poly_arr = [np.asarray(p) for p in polys]
        areas = [np.abs(poly_area(p[:,0], p[:,1])) for p in poly_arr]
        poly_ctr = [p.mean(axis=0) for p in poly_arr]
    
        sub_polys = []
        for poly, area, pctr in zip(polys, areas, poly_ctr):
            quads, quad_x_idx, quad_y_idx = self.mesh.quads_nearest(pctr)
            nq = quads.shape[0] * quads.shape[1]
            quads.shape = (nq, 4, 2) # so that we can do "for q in quads"
            clip_polys = [np.asarray(cp, dtype='f8').squeeze() for cp in clip_polys_by_one_poly(quads, poly) if len(cp) > 0]
            frac_areas = [np.abs(poly_area(p[:,0], p[:,1])/area) for p in clip_polys]
            sub_polys.append((clip_polys, frac_areas))
        return sub_polys    


    # frac_areas = [np.abs(pyclipper.Area(cp)/area) for cp in clip_polys]
        
        
