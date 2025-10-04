"""
Comprehensive analysis of oceanographic datasets
"""

import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from glob import glob
from datetime import datetime

def analyze_chl_datasets():
    """Analyze chlorophyll datasets"""
    print("=" * 80)
    print("CHLOROPHYLL (CHL) DATASET ANALYSIS")
    print("=" * 80)
    
    chl_files = glob('chl/*.nc')
    print(f"\nTotal files: {len(chl_files)}")
    
    if chl_files:
        # Analyze first file in detail
        sample_file = chl_files[0]
        print(f"\nDetailed analysis of: {os.path.basename(sample_file)}")
        
        with nc.Dataset(sample_file) as ds:
            print("\n--- Dimensions ---")
            for dim_name, dim in ds.dimensions.items():
                print(f"  {dim_name}: {len(dim)}")
            
            print("\n--- Variables ---")
            for var_name in ds.variables.keys():
                var = ds.variables[var_name]
                print(f"  {var_name}: {var.shape} - {var.dtype}")
                if hasattr(var, 'long_name'):
                    print(f"    Long name: {var.long_name}")
                if hasattr(var, 'units'):
                    print(f"    Units: {var.units}")
            
            print("\n--- Key Global Attributes ---")
            attrs_to_show = ['title', 'product_name', 'time_coverage_start', 
                            'time_coverage_end', 'geospatial_lat_min', 
                            'geospatial_lat_max', 'geospatial_lon_min', 
                            'geospatial_lon_max', 'spatialResolution']
            for attr in attrs_to_show:
                if hasattr(ds, attr):
                    print(f"  {attr}: {getattr(ds, attr)}")
            
            # Analyze chlor_a data
            if 'chlor_a' in ds.variables:
                chlor_data = ds.variables['chlor_a'][:]
                print("\n--- Chlorophyll-a Statistics ---")
                print(f"  Shape: {chlor_data.shape}")
                print(f"  Valid data points: {np.sum(~chlor_data.mask) if hasattr(chlor_data, 'mask') else chlor_data.size}")
                print(f"  Missing data points: {np.sum(chlor_data.mask) if hasattr(chlor_data, 'mask') else 0}")
                valid_data = chlor_data[~chlor_data.mask] if hasattr(chlor_data, 'mask') else chlor_data
                if valid_data.size > 0:
                    print(f"  Min: {np.min(valid_data):.6f}")
                    print(f"  Max: {np.max(valid_data):.6f}")
                    print(f"  Mean: {np.mean(valid_data):.6f}")
                    print(f"  Median: {np.median(valid_data):.6f}")
        
        # Summary of all files
        print("\n--- All CHL Files ---")
        for f in sorted(chl_files):
            basename = os.path.basename(f)
            # Extract date from filename
            parts = basename.split('.')
            if len(parts) > 1:
                date_str = parts[1]
                resolution = '4km' if '4km' in basename else '9km'
                print(f"  {basename[:50]:50s} Date: {date_str}, Resolution: {resolution}")

def analyze_sst_datasets():
    """Analyze sea surface temperature datasets"""
    print("\n" + "=" * 80)
    print("SEA SURFACE TEMPERATURE (SST) DATASET ANALYSIS")
    print("=" * 80)
    
    sst_files = glob('sst/*.nc')
    print(f"\nTotal files: {len(sst_files)}")
    
    if sst_files:
        # Analyze first file in detail
        sample_file = [f for f in sst_files if '(1)' not in f][0]
        print(f"\nDetailed analysis of: {os.path.basename(sample_file)}")
        
        with nc.Dataset(sample_file) as ds:
            print("\n--- Dimensions ---")
            for dim_name, dim in ds.dimensions.items():
                print(f"  {dim_name}: {len(dim)}")
            
            print("\n--- Variables (first 15) ---")
            for i, var_name in enumerate(list(ds.variables.keys())[:15]):
                var = ds.variables[var_name]
                print(f"  {var_name}: {var.shape} - {var.dtype}")
                if hasattr(var, 'long_name'):
                    print(f"    {var.long_name[:80]}")
            
            print("\n--- Key Global Attributes ---")
            attrs_to_show = ['title', 'start_time', 'stop_time', 
                            'northernmost_latitude', 'southernmost_latitude',
                            'westernmost_longitude', 'easternmost_longitude',
                            'spatial_resolution', 'platform']
            for attr in attrs_to_show:
                if hasattr(ds, attr):
                    val = getattr(ds, attr)
                    print(f"  {attr}: {val[:100] if isinstance(val, str) and len(val) > 100 else val}")
            
            # Analyze SST data
            if 'sea_surface_temperature' in ds.variables:
                sst_data = ds.variables['sea_surface_temperature'][:]
                print("\n--- SST Statistics ---")
                print(f"  Shape: {sst_data.shape}")
                valid_data = sst_data[~sst_data.mask] if hasattr(sst_data, 'mask') else sst_data
                if valid_data.size > 0:
                    print(f"  Valid points: {valid_data.size}")
                    print(f"  Min: {np.min(valid_data):.2f} K")
                    print(f"  Max: {np.max(valid_data):.2f} K")
                    print(f"  Mean: {np.mean(valid_data):.2f} K")
        
        # Summary of all files
        print("\n--- All SST Files ---")
        unique_files = [f for f in sorted(sst_files) if '(1)' not in f]
        for f in unique_files:
            basename = os.path.basename(f)
            # Extract datetime from filename
            datetime_str = basename[:14]
            print(f"  {basename[:60]:60s} Time: {datetime_str}")

def analyze_wave_datasets():
    """Analyze SWOT SSH/wave datasets"""
    print("\n" + "=" * 80)
    print("WAVE/SSH DATASET ANALYSIS (SWOT)")
    print("=" * 80)
    
    wave_files = glob('waves/*.nc')
    print(f"\nTotal files: {len(wave_files)}")
    
    if wave_files:
        # Analyze first file in detail
        sample_file = [f for f in wave_files if 'Basic' in f][0]
        print(f"\nDetailed analysis of: {os.path.basename(sample_file)}")
        
        with nc.Dataset(sample_file) as ds:
            print("\n--- Dimensions ---")
            for dim_name, dim in ds.dimensions.items():
                print(f"  {dim_name}: {len(dim)}")
            
            print("\n--- Variables (first 20) ---")
            for i, var_name in enumerate(list(ds.variables.keys())[:20]):
                var = ds.variables[var_name]
                print(f"  {var_name}: {var.shape} - {var.dtype}")
                if hasattr(var, 'long_name'):
                    print(f"    {var.long_name[:80]}")
            
            print("\n--- Key Global Attributes ---")
            attrs_to_show = ['title', 'mission_name', 'time_coverage_start', 
                            'time_coverage_end', 'geospatial_lat_min',
                            'geospatial_lat_max', 'geospatial_lon_min',
                            'geospatial_lon_max']
            for attr in attrs_to_show:
                if hasattr(ds, attr):
                    val = getattr(ds, attr)
                    print(f"  {attr}: {val[:100] if isinstance(val, str) and len(val) > 100 else val}")
        
        # Summary of all files
        print("\n--- All Wave/SSH Files ---")
        for f in sorted(wave_files):
            basename = os.path.basename(f)
            file_type = 'Basic' if 'Basic' in basename else 'Expert' if 'Expert' in basename else 'WindWave' if 'WindWave' in basename else 'Unsmoothed'
            print(f"  {basename[:70]:70s}")
            print(f"    Type: {file_type}")

def analyze_spottings_data():
    """Analyze species spottings/occurrence data"""
    print("\n" + "=" * 80)
    print("SPECIES SPOTTINGS/OCCURRENCE DATA ANALYSIS")
    print("=" * 80)
    
    spotting_dirs = glob('spottings/*')
    print(f"\nTotal subdirectories: {len(spotting_dirs)}")
    
    if spotting_dirs:
        # Analyze first directory
        sample_dir = spotting_dirs[0]
        print(f"\nDetailed analysis of: {os.path.basename(sample_dir)}")
        
        # List files in directory
        files = os.listdir(sample_dir)
        print(f"\nFiles in directory: {files}")
        
        # Analyze occurrence.txt if it exists
        occurrence_file = os.path.join(sample_dir, 'occurrence.txt')
        if os.path.exists(occurrence_file):
            print("\n--- Occurrence Data ---")
            df = pd.read_csv(occurrence_file, sep='\t', nrows=5)
            print(f"  Columns ({len(df.columns)}): {', '.join(df.columns[:10])}...")
            print(f"\n  First few rows:")
            print(df.head())
            
            # Get full dataset stats
            df_full = pd.read_csv(occurrence_file, sep='\t')
            print(f"\n  Total records: {len(df_full)}")
            
            if 'species' in df_full.columns:
                unique_species = df_full['species'].nunique()
                print(f"  Unique species: {unique_species}")
                print(f"\n  Top 5 species:")
                print(df_full['species'].value_counts().head())
            
            if 'decimalLatitude' in df_full.columns and 'decimalLongitude' in df_full.columns:
                print(f"\n  Geographic extent:")
                print(f"    Latitude: {df_full['decimalLatitude'].min():.2f} to {df_full['decimalLatitude'].max():.2f}")
                print(f"    Longitude: {df_full['decimalLongitude'].min():.2f} to {df_full['decimalLongitude'].max():.2f}")
            
            if 'eventDate' in df_full.columns:
                print(f"\n  Temporal extent:")
                dates = pd.to_datetime(df_full['eventDate'], errors='coerce')
                print(f"    From: {dates.min()}")
                print(f"    To: {dates.max()}")
        
        print("\n--- All Spotting Directories ---")
        for d in sorted(spotting_dirs):
            basename = os.path.basename(d)
            print(f"  {basename}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OCEANOGRAPHIC DATASET ANALYSIS")
    print("Workspace: d:\\sharkxx")
    print("=" * 80)
    
    analyze_chl_datasets()
    analyze_sst_datasets()
    analyze_wave_datasets()
    analyze_spottings_data()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
