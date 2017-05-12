"""
This set of tests reads the GLM data two ways, one by applying the unsigned integer conversion
manually, and the other by using the automatic method implemented in the library.

It was used to test PR #658 developed in response to issue #656 on the unidata/netcdf4-python library.
The second, full-auto method should work if the version (>=1.2.8) of netcdf4-python post-dates this PR.

These tests were developed with GLM data dating after 24 April 2017, but may not work with
later production upgrades if the unsigned int encoding method used in the production system changes.
"""


filename = '/data/LCFA-production/OR_GLM-L2-LCFA_G16_s20171161230400_e20171161231000_c20171161231027.nc'
some_flash = 6359

import netCDF4
nc = netCDF4.Dataset(filename)

event_lons = nc.variables['event_lon']
event_lons.set_auto_scale(False)
scale_factor = event_lons.scale_factor
add_offset = event_lons.add_offset

event_lons = event_lons[:].astype('u2')
event_lons_fixed = (event_lons[:])*scale_factor+add_offset

nc.close()

print("Manual scaling")
print(event_lons_fixed.min())
print(event_lons_fixed.max())


# lon_fov = (-156.06, -22.94)
# dlon_fov = lon_fov[1]-lon_fov[0]
# lat_fov = (-66.56, 66.56)
# scale_factor = 0.00203128 # from file and spec; same for both


# ------

filename = '/data/LCFA-production/OR_GLM-L2-LCFA_G16_s20171161230400_e20171161231000_c20171161231027.nc'
some_flash = 6359

import netCDF4
nc = netCDF4.Dataset(filename)

event_lons = nc.variables['event_lon']
event_lons_fixed = event_lons[:]

nc.close()

print("Auto scaling")
print(event_lons_fixed.min())
print(event_lons_fixed.max())
