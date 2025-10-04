"""
Quick Comparison: Regional vs Global Analysis
Shows the key differences in data extent and parameter selection
"""

import pandas as pd
import numpy as np
from glob import glob

def compare_regional_vs_global():
    """Compare what changes between regional and global analysis"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "REGIONAL vs GLOBAL COMPARISON" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Load shark data
    occurrence_files = glob('spottings/*/occurrence.txt')
    all_data = []
    for file in occurrence_files:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        all_data.append(df)
    
    shark_data = pd.concat(all_data, ignore_index=True)
    
    # Filter for sharks only
    shark_keywords = ['shark', 'carcharodon', 'prionace', 'notorynchus']
    shark_mask = shark_data['species'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in shark_keywords)
    )
    shark_data = shark_data[shark_mask].copy()
    
    # Add coordinates
    shark_data['latitude'] = shark_data['decimalLatitude']
    shark_data['longitude'] = shark_data['decimalLongitude']
    
    # Remove missing coordinates
    shark_data = shark_data.dropna(subset=['latitude', 'longitude'])
    
    print("=" * 80)
    print("FULL DATASET (GLOBAL)")
    print("=" * 80)
    
    lat_min_global = shark_data['latitude'].min()
    lat_max_global = shark_data['latitude'].max()
    lon_min_global = shark_data['longitude'].min()
    lon_max_global = shark_data['longitude'].max()
    lat_range_global = lat_max_global - lat_min_global
    lon_range_global = lon_max_global - lon_min_global
    
    print(f"\nTotal observations: {len(shark_data):,}")
    print(f"\nGeographic extent:")
    print(f"  Latitude:  {lat_min_global:7.2f}° to {lat_max_global:7.2f}° (range: {lat_range_global:6.2f}°)")
    print(f"  Longitude: {lon_min_global:7.2f}° to {lon_max_global:7.2f}° (range: {lon_range_global:6.2f}°)")
    
    # Determine scale
    if lat_range_global > 50 or lon_range_global > 100:
        scale_global = "GLOBAL"
        eps_global = 2.0
        kde_global = 1.0
        zoom_global = 2
        grid_global = 100
    elif lat_range_global > 20 or lon_range_global > 40:
        scale_global = "REGIONAL"
        eps_global = 1.0
        kde_global = 0.5
        zoom_global = 4
        grid_global = 75
    else:
        scale_global = "LOCAL"
        eps_global = 0.5
        kde_global = 0.3
        zoom_global = 7
        grid_global = 50
    
    print(f"\n🌍 Detected scale: {scale_global}")
    print(f"\nAutomatically selected parameters:")
    print(f"  DBSCAN eps:      {eps_global}°")
    print(f"  KDE bandwidth:   {kde_global}°")
    print(f"  Map zoom:        {zoom_global}")
    print(f"  Prediction grid: {grid_global}×{grid_global} cells")
    
    # Species breakdown
    print(f"\nSpecies distribution:")
    species_counts = shark_data['species'].value_counts()
    for species, count in species_counts.head(10).items():
        pct = 100 * count / len(shark_data)
        print(f"  {species:40s}: {count:6,} ({pct:5.1f}%)")
    
    if len(species_counts) > 10:
        print(f"  ... and {len(species_counts) - 10} more species")
    
    # Regional filter (North America example)
    print("\n" + "=" * 80)
    print("REGIONAL SUBSET (North America)")
    print("=" * 80)
    
    # North America bounds (rough)
    NA_lat_min, NA_lat_max = 24, 70
    NA_lon_min, NA_lon_max = -170, -50
    
    regional_data = shark_data[
        (shark_data['latitude'] >= NA_lat_min) &
        (shark_data['latitude'] <= NA_lat_max) &
        (shark_data['longitude'] >= NA_lon_min) &
        (shark_data['longitude'] <= NA_lon_max)
    ].copy()
    
    print(f"\nFiltered to North America bounds:")
    print(f"  Latitude:  {NA_lat_min}° to {NA_lat_max}°")
    print(f"  Longitude: {NA_lon_min}° to {NA_lon_max}°")
    
    print(f"\nObservations: {len(regional_data):,} ({100*len(regional_data)/len(shark_data):.1f}% of global)")
    
    if len(regional_data) > 0:
        lat_min_regional = regional_data['latitude'].min()
        lat_max_regional = regional_data['latitude'].max()
        lon_min_regional = regional_data['longitude'].min()
        lon_max_regional = regional_data['longitude'].max()
        lat_range_regional = lat_max_regional - lat_min_regional
        lon_range_regional = lon_max_regional - lon_min_regional
        
        print(f"\nActual extent:")
        print(f"  Latitude:  {lat_min_regional:7.2f}° to {lat_max_regional:7.2f}° (range: {lat_range_regional:6.2f}°)")
        print(f"  Longitude: {lon_min_regional:7.2f}° to {lon_max_regional:7.2f}° (range: {lon_range_regional:6.2f}°)")
        
        # Fixed parameters for regional
        eps_regional = 0.5
        kde_regional = 0.3
        zoom_regional = 4
        grid_regional = 50
        
        print(f"\n🗺️  Fixed regional parameters:")
        print(f"  DBSCAN eps:      {eps_regional}°")
        print(f"  KDE bandwidth:   {kde_regional}°")
        print(f"  Map zoom:        {zoom_regional}")
        print(f"  Prediction grid: {grid_regional}×{grid_regional} cells")
        
        print(f"\nSpecies distribution:")
        species_counts_regional = regional_data['species'].value_counts()
        for species, count in species_counts_regional.head(5).items():
            pct = 100 * count / len(regional_data)
            print(f"  {species:40s}: {count:6,} ({pct:5.1f}%)")
    
    # Summary comparison
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 28 + "KEY DIFFERENCES" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    comparison = pd.DataFrame({
        'Parameter': [
            'Geographic Filter',
            'Observations',
            'Scale Detection',
            'DBSCAN eps',
            'KDE bandwidth',
            'Map zoom',
            'Grid size',
            'Map center'
        ],
        'Regional (main_shark_analysis.py)': [
            'North America only',
            f"{len(regional_data):,}",
            'Fixed (not adaptive)',
            '0.5° (fixed)',
            '0.3° (fixed)',
            '4 (fixed)',
            '50×50 (fixed)',
            'North America centroid'
        ],
        'Global (main_global_analysis.py)': [
            'Worldwide (no filter)',
            f"{len(shark_data):,}",
            f'Adaptive ({scale_global})',
            f'{eps_global}° (adaptive)',
            f'{kde_global}° (adaptive)',
            f'{zoom_global} (adaptive)',
            f'{grid_global}×{grid_global} (adaptive)',
            'Data-driven median'
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n📌 Use REGIONAL script when:")
    print("  ✓ You know data is North America only")
    print("  ✓ You want fixed parameters for reproducibility")
    print("  ✓ You're comparing against previous North America studies")
    
    print("\n🌍 Use GLOBAL script when:")
    print("  ✓ Data spans multiple continents/ocean basins")
    print("  ✓ You want automatic parameter optimization")
    print("  ✓ You're analyzing different species with different ranges")
    print("  ✓ You don't know the geographic extent beforehand")
    
    print("\n💡 For your current dataset:")
    if lat_range_global > 50 or lon_range_global > 100:
        print("  → RECOMMENDED: Use GLOBAL script (main_global_analysis.py)")
        print("    Your data is truly global in extent!")
    elif len(regional_data) / len(shark_data) > 0.8:
        print("  → RECOMMENDED: Use REGIONAL script (main_shark_analysis.py)")
        print("    Most of your data is in North America already!")
    else:
        print("  → RECOMMENDED: Use GLOBAL script (main_global_analysis.py)")
        print("    Your data spans multiple regions beyond North America!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        compare_regional_vs_global()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
