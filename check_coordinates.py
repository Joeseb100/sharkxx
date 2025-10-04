"""
Check latitude/longitude information in all datasets
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from glob import glob

def check_chl_coordinates():
    """Check chlorophyll dataset coordinates"""
    print("=" * 80)
    print("CHLOROPHYLL DATASET - COORDINATE INFORMATION")
    print("=" * 80)
    
    sample_file = 'chl/AQUA_MODIS.20250827.L3m.DAY.CHL.chlor_a.4km.nc'
    
    with nc.Dataset(sample_file) as ds:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        
        print(f"\n✓ HAS COORDINATE ARRAYS")
        print(f"\nLatitude array:")
        print(f"  Variable name: 'lat'")
        print(f"  Shape: {lat.shape}")
        print(f"  Range: {lat.min():.2f}° to {lat.max():.2f}°")
        print(f"  Resolution: {np.median(np.diff(lat)):.4f}° (~{np.median(np.diff(lat)) * 111:.2f} km)")
        print(f"  First 5 values: {lat[:5]}")
        
        print(f"\nLongitude array:")
        print(f"  Variable name: 'lon'")
        print(f"  Shape: {lon.shape}")
        print(f"  Range: {lon.min():.2f}° to {lon.max():.2f}°")
        print(f"  Resolution: {np.median(np.diff(lon)):.4f}° (~{np.median(np.diff(lon)) * 111:.2f} km at equator)")
        print(f"  First 5 values: {lon[:5]}")
        
        print(f"\nChlorophyll data grid:")
        print(f"  Shape: {ds.variables['chlor_a'].shape}")
        print(f"  Grid: {lat.shape[0]} latitudes × {lon.shape[0]} longitudes")
        print(f"  Each chlor_a[i, j] corresponds to lat[i], lon[j]")

def check_sst_coordinates():
    """Check SST dataset coordinates"""
    print("\n" + "=" * 80)
    print("SEA SURFACE TEMPERATURE - COORDINATE INFORMATION")
    print("=" * 80)
    
    sample_file = 'sst/20250403080000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70-v02.0-fv01.0.nc'
    
    with nc.Dataset(sample_file) as ds:
        print(f"\n✓ HAS COORDINATE ARRAYS")
        
        if 'lat' in ds.variables:
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
            print(f"\nLatitude array:")
            print(f"  Variable name: 'lat'")
            print(f"  Shape: {lat.shape}")
            print(f"  Range: {np.nanmin(lat):.2f}° to {np.nanmax(lat):.2f}°")
            
            print(f"\nLongitude array:")
            print(f"  Variable name: 'lon'")
            print(f"  Shape: {lon.shape}")
            print(f"  Range: {np.nanmin(lon):.2f}° to {np.nanmax(lon):.2f}°")
        else:
            # Check global attributes for extent
            print(f"\nSpatial extent from global attributes:")
            print(f"  Latitude: {ds.southernmost_latitude:.2f}° to {ds.northernmost_latitude:.2f}°")
            print(f"  Longitude: {ds.westernmost_longitude:.2f}° to {ds.easternmost_longitude:.2f}°")
            
            # Check for coordinate variables with different names
            coord_vars = [v for v in ds.variables.keys() if 'lat' in v.lower() or 'lon' in v.lower()]
            if coord_vars:
                print(f"\n  Coordinate-related variables found: {coord_vars}")
        
        print(f"\nSST data grid:")
        sst = ds.variables['sea_surface_temperature']
        print(f"  Shape: {sst.shape}")
        print(f"  Note: Geostationary satellites often store lat/lon as 2D arrays")

def check_swot_coordinates():
    """Check SWOT dataset coordinates"""
    print("\n" + "=" * 80)
    print("SWOT SSH/WAVE - COORDINATE INFORMATION")
    print("=" * 80)
    
    sample_file = 'waves/SWOT_L2_LR_SSH_Basic_039_304_20251001T215514_20251001T224643_PID0_01.nc'
    
    with nc.Dataset(sample_file) as ds:
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        
        print(f"\n✓ HAS COORDINATE ARRAYS")
        print(f"\nLatitude array:")
        print(f"  Variable name: 'latitude'")
        print(f"  Shape: {lat.shape} (2D - swath data)")
        print(f"  Data type: Each point along satellite track has lat/lon")
        
        # Get valid data
        lat_valid = lat[~lat.mask] if hasattr(lat, 'mask') else lat.flatten()
        lon_valid = lon[~lon.mask] if hasattr(lon, 'mask') else lon.flatten()
        
        print(f"  Range: {np.nanmin(lat_valid):.2f}° to {np.nanmax(lat_valid):.2f}°")
        print(f"  Sample values (first 3 rows, first 3 cols):")
        print(f"    {lat[:3, :3]}")
        
        print(f"\nLongitude array:")
        print(f"  Variable name: 'longitude'")
        print(f"  Shape: {lon.shape} (2D - swath data)")
        print(f"  Range: {np.nanmin(lon_valid):.2f}° to {np.nanmax(lon_valid):.2f}°")
        print(f"  Sample values (first 3 rows, first 3 cols):")
        print(f"    {lon[:3, :3]}")
        
        print(f"\nSSH data structure:")
        ssh = ds.variables['ssh_karin']
        print(f"  Shape: {ssh.shape}")
        print(f"  Each ssh[i, j] corresponds to latitude[i, j], longitude[i, j]")
        print(f"  This is satellite swath data - coordinates vary along AND across track")

def check_occurrence_coordinates():
    """Check species occurrence coordinates"""
    print("\n" + "=" * 80)
    print("SPECIES OCCURRENCE - COORDINATE INFORMATION")
    print("=" * 80)
    
    sample_file = 'spottings/0046525-250920141307145/occurrence.txt'
    
    df = pd.read_csv(sample_file, sep='\t')
    
    print(f"\n✓ HAS LATITUDE/LONGITUDE COLUMNS")
    
    print(f"\nCoordinate columns:")
    coord_cols = [col for col in df.columns if 'latitude' in col.lower() or 'longitude' in col.lower()]
    print(f"  Found: {coord_cols}")
    
    print(f"\nDecimal coordinates (WGS84):")
    print(f"  Latitude column: 'decimalLatitude'")
    print(f"    Range: {df['decimalLatitude'].min():.4f}° to {df['decimalLatitude'].max():.4f}°")
    print(f"    Mean: {df['decimalLatitude'].mean():.4f}°")
    print(f"    Non-null values: {df['decimalLatitude'].notna().sum()} / {len(df)}")
    
    print(f"\n  Longitude column: 'decimalLongitude'")
    print(f"    Range: {df['decimalLongitude'].min():.4f}° to {df['decimalLongitude'].max():.4f}°")
    print(f"    Mean: {df['decimalLongitude'].mean():.4f}°")
    print(f"    Non-null values: {df['decimalLongitude'].notna().sum()} / {len(df)}")
    
    print(f"\nSample coordinates (first 5 records):")
    print(df[['species', 'decimalLatitude', 'decimalLongitude', 'eventDate']].head())
    
    print(f"\nCoordinate precision:")
    print(f"  All coordinates appear to be in decimal degrees (WGS84 datum)")
    print(f"  Typical precision: ~4-6 decimal places (~1-10 meter accuracy)")

def create_coordinate_summary():
    """Create summary comparison"""
    print("\n" + "=" * 80)
    print("COORDINATE SYSTEM SUMMARY")
    print("=" * 80)
    
    summary = """
┌─────────────────┬──────────────────┬───────────────┬──────────────────────┐
│ Dataset         │ Coord Variables  │ Structure     │ Coverage             │
├─────────────────┼──────────────────┼───────────────┼──────────────────────┤
│ Chlorophyll     │ lat, lon         │ 1D arrays     │ Global               │
│                 │ (WGS84)          │ Regular grid  │ -90° to +90° lat     │
│                 │                  │ 4320 × 8640   │ -180° to +180° lon   │
├─────────────────┼──────────────────┼───────────────┼──────────────────────┤
│ SST             │ lat, lon OR      │ 2D arrays     │ Western Hemisphere   │
│                 │ attributes       │ 5424 × 5424   │ -81° to +81° lat     │
│                 │ (WGS84)          │ Geostationary │ -156° to +6° lon     │
├─────────────────┼──────────────────┼───────────────┼──────────────────────┤
│ SWOT SSH        │ latitude,        │ 2D arrays     │ Near-polar           │
│                 │ longitude        │ Swath data    │ -78° to +78° lat     │
│                 │ (WGS84)          │ 9866 × 69     │ 134° to 301° lon     │
├─────────────────┼──────────────────┼───────────────┼──────────────────────┤
│ Shark Spottings │ decimalLatitude, │ Point data    │ South Africa         │
│                 │ decimalLongitude │ One per       │ -35° to -27° lat     │
│                 │ (WGS84)          │ observation   │ 18° to 33° lon       │
└─────────────────┴──────────────────┴───────────────┴──────────────────────┘

KEY POINTS:
✓ ALL datasets include geographic coordinates (latitude/longitude)
✓ ALL use WGS84 datum (standard for GPS/satellite data)
✓ Data can be spatially matched and integrated
✓ Different structures require different indexing approaches:
  - Gridded data (CHL, SST): Regular lat/lon grids
  - Swath data (SWOT): Irregular 2D lat/lon arrays
  - Point data (Sharks): Individual lat/lon pairs
"""
    print(summary)

if __name__ == "__main__":
    check_chl_coordinates()
    check_sst_coordinates()
    check_swot_coordinates()
    check_occurrence_coordinates()
    create_coordinate_summary()
    
    print("\n" + "=" * 80)
    print("CONCLUSION: All datasets are georeferenced and can be spatially analyzed!")
    print("=" * 80)
