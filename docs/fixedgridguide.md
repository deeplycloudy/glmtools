## Fixed grid coordinates: an introduction

The GLM grids are produced in the GOES fixed grid coordinates. This is the standard coordinate reference frame for the ABI observations, and for other geostationary satellite observations such as those from older GOES imagers or MSG SEVIRI.

The advantage of using the GLM data in fixed grid coordinates is that they overlay directly on other satellite obserations with no parallax. They will have the same parallax with respect to ground as any other satellite observations.

The units of fixed grid coordinates are radians: an angular measure. What angle is being measured? Picture yourself at the satellite, looking straight down (nadir) at the earth. An observation along this line away from the satellite has fixed grid position `(x = 0 rad, y = 0 rad)`. An observation at any other location will have a non-zero fixed grid location. This could be some other point on the earth, or a even a distant star. The fixed grid coordinates define that observation direction as the angle east and the angle north `(x, y)` from nadir. South and west are negative angles. The edge of the earth is about `x = 8.5° = 0.15 rad` to the east of nadir, and is also about `y = 8.5° = 0.15 rad` to the north of nadir.

This small [interactive 3D visualization](https://poly.google.com/view/90l_J_l28o3) gives a representation of the projection of fixed grid coordinates onto the earth.  In the visualization, the bright white line is nadir, blue lines are fixed grid coordinates, and green lines are Earth lat/long.

The [GOES-R Product Users' Guide](https://www.goes-r.gov/resources/docs.html#user) (see Vol. 3, L1b) further provides an excellent mathematical and visual introduction to fixed grid coordinates for the ABI instrument. The same introduction applies to the GLM grids produced by `glmtools`.

## Converting fixed grid coordinates to longitude/latitude

To convert fixed grid coordinates to latitude and longitude, it is necessary to assume a shape of the earth (for GOES observations this is basically the GRS80 ellipsoid), and then calculate the intersection point of the ray connecting the satellite to the Earth's surface.

Once you have installed glmtools there are helper routines you can can use to make this conversion. The GLM grids include a 1-dimensional array of `x` coordinates and a 1D array of `y` coordinates. When converted to `(lon, lat)`, the coordinate data will no longer be regular, so the final result is a 2D array of `(lon, lat)` coordinates.

The example below shows how to read in a time series of GLM 1 min grids (which aggregates them into a common dataset and adds a time dimension) and then calculates the 2D `lon` and `lat` arrays.
 
```
# Load the GLM data
glm = open_glm_time_series(['~/glmtools/glmtools/test/data/conus/2018/Jul/02/OR_GLM-L2-GLMC-M3_G16_s20181830433000_e20181830434000_c20182551446.nc'])
x_1d = glm.x
y_1d = glm.y

# Convert the 1D fixed grid coordinates to 2D lon, lat

from lmatools.grid.fixed import get_GOESR_coordsys
x,y = np.meshgrid(x_1d, y_1d) # Two 2D arrays of fixed grid coordinates
nadir = -75.0
geofixCS, grs80lla = get_GOESR_coordsys(nadir)
z=np.zeros_like(x)
lon,lat,alt=grs80lla.fromECEF(*geofixCS.toECEF(x,y,z))
lon.shape = x.shape
lat.shape = y.shape

# Add the 2D arrays back to the original dataset and save to disk. 
# This doesn't have the CF standard name metadata or unit information,
# but that isn't too hard to add.

import xarray as xr
glm['longitude'] = xr.DataArray(lon, dims=('y', 'x'))
glm['latitude'] = xr.DataArray(lat, dims=('y', 'x'))

glm.to_netcdf('glm_aggregate.nc')
```
