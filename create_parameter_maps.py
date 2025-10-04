"""
Create Multi-Parameter Interactive Maps
Generates maps with separate toggleable layers for each environmental parameter
"""

import pandas as pd
import numpy as np
from glob import glob
from multi_parameter_map import create_complete_parameter_maps
import warnings
warnings.filterwarnings('ignore')

def generate_multi_parameter_maps():
    """Generate comprehensive multi-parameter maps"""
    
    print("=" * 80)
    print("GENERATING MULTI-PARAMETER INTERACTIVE MAPS")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: LOAD AND PREPARE SHARK DATA
    # =========================================================================
    print("\n📊 Loading shark occurrence data...")
    
    occurrence_files = glob('spottings/*/occurrence.txt')
    all_data = []
    for file in occurrence_files:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        all_data.append(df)
    
    shark_data = pd.concat(all_data, ignore_index=True)
    
    # Filter for sharks
    shark_keywords = ['shark', 'carcharodon', 'prionace', 'notorynchus']
    shark_mask = shark_data['species'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in shark_keywords)
    )
    shark_data = shark_data[shark_mask].copy()
    
    # Clean coordinates
    shark_data = shark_data.dropna(subset=['decimalLatitude', 'decimalLongitude'])
    
    # Extract basic features
    shark_data['latitude'] = shark_data['decimalLatitude']
    shark_data['longitude'] = shark_data['decimalLongitude']
    
    # Add temporal features
    shark_data['date'] = pd.to_datetime(shark_data['eventDate'], errors='coerce')
    shark_data['month'] = shark_data['date'].dt.month
    shark_data['day_of_year'] = shark_data['date'].dt.dayofyear
    
    print(f"✓ Loaded {len(shark_data):,} shark observations")
    print(f"  Species: {shark_data['species'].nunique()}")
    print(f"  Geographic extent: {shark_data['latitude'].min():.2f}° to {shark_data['latitude'].max():.2f}° lat")
    print(f"                     {shark_data['longitude'].min():.2f}° to {shark_data['longitude'].max():.2f}° lon")
    
    # =========================================================================
    # STEP 2: ADD SIMULATED ENVIRONMENTAL PARAMETERS
    # =========================================================================
    print("\n🌡️ Generating environmental parameters...")
    
    # SST (Sea Surface Temperature)
    base_sst = 18 - (np.abs(shark_data['latitude']) - 30) * 0.5
    seasonal_effect = shark_data['month'].fillna(6).apply(
        lambda m: 3 * np.sin(2 * np.pi * (m - 3) / 12)
    )
    shark_data['sst'] = base_sst + seasonal_effect + np.random.normal(0, 1, len(shark_data))
    shark_data['sst'] = shark_data['sst'].clip(10, 30)
    
    # Chlorophyll-a
    coastal_distance = np.abs(shark_data['longitude'] - shark_data['longitude'].median())
    shark_data['chlorophyll'] = 1.5 * np.exp(-coastal_distance / 10) + np.random.uniform(0.05, 0.3, len(shark_data))
    shark_data['chlorophyll'] = shark_data['chlorophyll'].clip(0.01, 3.0)
    
    # Bathymetry (depth)
    shark_data['depth'] = 50 + coastal_distance * 15 + np.random.uniform(0, 200, len(shark_data))
    shark_data['depth'] = shark_data['depth'].clip(0, 5000)
    
    # Distance to coast
    shark_data['distance_to_coast'] = coastal_distance * 8 + np.random.uniform(0, 50, len(shark_data))
    shark_data['distance_to_coast'] = shark_data['distance_to_coast'].clip(0, 1000)
    
    # SST gradient (fronts)
    shark_data['sst_gradient'] = np.abs(np.random.normal(0, 0.3, len(shark_data)))
    
    print("✓ Environmental parameters generated:")
    print(f"  SST: {shark_data['sst'].min():.1f}°C to {shark_data['sst'].max():.1f}°C")
    print(f"  Chlorophyll: {shark_data['chlorophyll'].min():.3f} to {shark_data['chlorophyll'].max():.3f} mg/m³")
    print(f"  Depth: {shark_data['depth'].min():.0f}m to {shark_data['depth'].max():.0f}m")
    print(f"  Distance to coast: {shark_data['distance_to_coast'].min():.0f}km to {shark_data['distance_to_coast'].max():.0f}km")
    print(f"  SST gradient: {shark_data['sst_gradient'].min():.3f} to {shark_data['sst_gradient'].max():.3f} °C/km")
    
    # =========================================================================
    # STEP 3: GENERATE HOTSPOTS
    # =========================================================================
    print("\n🔥 Detecting hotspots...")
    
    from sklearn.cluster import DBSCAN
    
    coords = np.column_stack([shark_data['longitude'], shark_data['latitude']])
    
    # Adaptive clustering
    lat_range = shark_data['latitude'].max() - shark_data['latitude'].min()
    lon_range = shark_data['longitude'].max() - shark_data['longitude'].min()
    
    if lat_range > 50 or lon_range > 100:
        eps = 2.0
    elif lat_range > 20 or lon_range > 40:
        eps = 1.0
    else:
        eps = 0.5
    
    min_samples = max(10, int(len(shark_data) * 0.01))
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(coords)
    
    # Create hotspot data
    hotspots = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_coords = coords[cluster_mask]
        
        if len(cluster_coords) > 0:
            hotspots.append({
                'cluster_id': cluster_id,
                'center_lat': cluster_coords[:, 1].mean(),
                'center_lon': cluster_coords[:, 0].mean(),
                'size': cluster_mask.sum()
            })
    
    hotspots = sorted(hotspots, key=lambda x: x['size'], reverse=True)
    
    print(f"✓ Detected {len(hotspots)} hotspots")
    if hotspots:
        print(f"  Largest hotspot: {hotspots[0]['size']} sharks at {hotspots[0]['center_lat']:.3f}°, {hotspots[0]['center_lon']:.3f}°")
    
    # =========================================================================
    # STEP 4: GENERATE PREDICTION GRID
    # =========================================================================
    print("\n📊 Creating prediction grid...")
    
    # Create grid for probability predictions
    lat_min, lat_max = shark_data['latitude'].min(), shark_data['latitude'].max()
    lon_min, lon_max = shark_data['longitude'].min(), shark_data['longitude'].max()
    
    # Expand by 5%
    lat_buffer = (lat_max - lat_min) * 0.05
    lon_buffer = (lon_max - lon_min) * 0.05
    
    grid_size = 50  # Moderate resolution for web performance
    
    lat_grid = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, grid_size)
    lon_grid = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, grid_size)
    
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    
    grid_data = pd.DataFrame({
        'latitude': lat_mesh.ravel(),
        'longitude': lon_mesh.ravel()
    })
    
    # Simulate habitat suitability probabilities
    # Higher probability near actual observations
    probabilities = np.zeros(len(grid_data))
    
    for i, (grid_lat, grid_lon) in enumerate(zip(grid_data['latitude'], grid_data['longitude'])):
        # Calculate distance to nearest observation
        distances = np.sqrt((shark_data['latitude'] - grid_lat)**2 + 
                           (shark_data['longitude'] - grid_lon)**2)
        min_distance = distances.min()
        
        # Probability decreases with distance
        probability = np.exp(-min_distance * 2)  # Exponential decay
        
        # Add environmental factors
        temp_factor = 1.0 if 15 <= grid_lat <= 25 else 0.5
        probability *= temp_factor
        
        probabilities[i] = probability
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.max()
    
    print(f"✓ Generated {grid_size}×{grid_size} prediction grid")
    print(f"  Probability range: {probabilities.min():.3f} to {probabilities.max():.3f}")
    print(f"  High suitability cells (>0.5): {(probabilities > 0.5).sum()}")
    
    # =========================================================================
    # STEP 5: CREATE MULTI-PARAMETER MAPS
    # =========================================================================
    print("\n🗺️ Creating multi-parameter interactive map...")
    
    filename = create_complete_parameter_maps(
        shark_data=shark_data,
        grid_data=grid_data,
        hotspots=hotspots,
        probabilities=probabilities
    )
    
    # =========================================================================
    # STEP 6: CREATE PARAMETER SUMMARY
    # =========================================================================
    print("\n📋 Creating parameter summary...")
    
    summary = {
        'parameter': [],
        'min_value': [],
        'max_value': [],
        'mean_value': [],
        'std_value': [],
        'units': [],
        'description': []
    }
    
    parameters = {
        'sst': ('Sea Surface Temperature', '°C', 'Water temperature at surface'),
        'chlorophyll': ('Chlorophyll-a', 'mg/m³', 'Primary productivity indicator'),
        'depth': ('Water Depth', 'm', 'Bathymetry/ocean depth'),
        'distance_to_coast': ('Distance to Coast', 'km', 'Distance from nearest coastline'),
        'sst_gradient': ('SST Gradient', '°C/km', 'Temperature fronts and upwelling'),
        'month': ('Month', 'month', 'Month of observation (1-12)'),
        'day_of_year': ('Day of Year', 'day', 'Julian day (1-365)')
    }
    
    for param, (name, unit, desc) in parameters.items():
        if param in shark_data.columns:
            values = shark_data[param].dropna()
            summary['parameter'].append(name)
            summary['min_value'].append(values.min())
            summary['max_value'].append(values.max())
            summary['mean_value'].append(values.mean())
            summary['std_value'].append(values.std())
            summary['units'].append(unit)
            summary['description'].append(desc)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('outputs/parameter_summary.csv', index=False)
    
    print("✓ Parameter summary saved: outputs/parameter_summary.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "MULTI-PARAMETER MAP COMPLETE!" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("📂 Files created:")
    print(f"  🗺️  Interactive map: {filename}")
    print(f"  📊 Parameter summary: outputs/parameter_summary.csv")
    
    print("\n🎛️ Map features:")
    print("  ✅ Toggleable layers for each parameter")
    print("  ✅ Multiple base map options")
    print("  ✅ Species-specific distributions")
    print("  ✅ Hotspot cluster markers")
    print("  ✅ Habitat suitability probability")
    print("  ✅ Environmental parameter overlays")
    
    print("\n🔧 Available parameters:")
    for _, (name, unit, desc) in parameters.items():
        print(f"  • {name} ({unit}): {desc}")
    
    print(f"\n📱 Usage:")
    print(f"  1. Open {filename} in web browser")
    print(f"  2. Use layer control panel to toggle parameters")
    print(f"  3. Click on points for detailed parameter values")
    print(f"  4. Switch between different base maps")
    
    return filename

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    try:
        map_file = generate_multi_parameter_maps()
        print(f"\n🎉 Success! Open {map_file} to explore your multi-parameter shark map!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()