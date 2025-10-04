"""
Enhanced Hotspot Analysis - Top 10 Zones with Detailed Statistics
"""

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def analyze_top_10_zones():
    """Analyze the top 10 shark zones with detailed statistics"""
    
    print("=" * 80)
    print("TOP 10 SHARK ZONES - DETAILED ANALYSIS")
    print("=" * 80)
    
    # Load shark data
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
    coords = np.column_stack([shark_data['decimalLongitude'], shark_data['decimalLatitude']])
    
    print(f"\nAnalyzing {len(shark_data):,} shark observations")
    
    # Apply clustering with adaptive parameters
    lat_range = shark_data['decimalLatitude'].max() - shark_data['decimalLatitude'].min()
    lon_range = shark_data['decimalLongitude'].max() - shark_data['decimalLongitude'].min()
    
    if lat_range > 50 or lon_range > 100:
        eps = 2.0  # Global scale
    elif lat_range > 20 or lon_range > 40:
        eps = 1.0  # Regional scale
    else:
        eps = 0.5  # Local scale
    
    min_samples = max(10, int(len(shark_data) * 0.01))  # At least 1% of data
    
    print(f"\nClustering parameters:")
    print(f"  eps: {eps}° (~{eps*111:.0f} km)")
    print(f"  min_samples: {min_samples}")
    
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(coords)
    
    # Analyze clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nClustering results:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    # Create detailed zone analysis
    zones = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_data = shark_data[cluster_mask].copy()
        cluster_coords = coords[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
        
        # Calculate zone statistics
        center_lon = cluster_coords[:, 0].mean()
        center_lat = cluster_coords[:, 1].mean()
        
        # Geographic extent
        lat_min, lat_max = cluster_data['decimalLatitude'].min(), cluster_data['decimalLatitude'].max()
        lon_min, lon_max = cluster_data['decimalLongitude'].min(), cluster_data['decimalLongitude'].max()
        
        # Calculate area (rough approximation)
        lat_km = (lat_max - lat_min) * 111
        lon_km = (lon_max - lon_min) * 111 * np.cos(np.radians(center_lat))
        area_km2 = lat_km * lon_km
        
        # Density
        density = len(cluster_data) / max(area_km2, 1)  # sharks per km²
        
        # Species breakdown
        species_counts = cluster_data['species'].value_counts()
        dominant_species = species_counts.index[0]
        species_diversity = len(species_counts)
        
        # Temporal analysis
        cluster_data['date'] = pd.to_datetime(cluster_data['eventDate'], errors='coerce')
        years = cluster_data['date'].dt.year.dropna()
        year_range = f"{years.min():.0f}-{years.max():.0f}" if len(years) > 0 else "Unknown"
        
        # Create zone description
        zone = {
            'zone_id': cluster_id + 1,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'total_sharks': len(cluster_data),
            'area_km2': area_km2,
            'density_per_km2': density,
            'lat_range': (lat_min, lat_max),
            'lon_range': (lon_min, lon_max),
            'dominant_species': dominant_species,
            'species_count': species_diversity,
            'year_range': year_range,
            'species_breakdown': dict(species_counts),
            'percentage_of_total': 100 * len(cluster_data) / len(shark_data)
        }
        
        zones.append(zone)
    
    # Sort by number of sharks
    zones = sorted(zones, key=lambda x: x['total_sharks'], reverse=True)
    
    # Display top 10 zones
    print("\n" + "=" * 120)
    print("TOP 10 SHARK ZONES - DETAILED BREAKDOWN")
    print("=" * 120)
    
    print(f"{'#':>2} {'Zone':>5} {'Location':>20} {'Sharks':>8} {'Area(km²)':>10} {'Density':>8} {'Species':>8} {'Years':>12} {'% Total':>8}")
    print("─" * 120)
    
    for i, zone in enumerate(zones[:10]):
        location = f"{zone['center_lat']:6.2f}, {zone['center_lon']:7.2f}"
        area_str = f"{zone['area_km2']:8.0f}" if zone['area_km2'] < 1e6 else f"{zone['area_km2']/1e6:6.1f}M"
        density_str = f"{zone['density_per_km2']:6.2f}" if zone['density_per_km2'] < 100 else f"{zone['density_per_km2']:6.0f}"
        
        print(f"{i+1:>2} {zone['zone_id']:>5} {location:>20} {zone['total_sharks']:>8,} "
              f"{area_str:>10} {density_str:>8} {zone['species_count']:>8} "
              f"{zone['year_range']:>12} {zone['percentage_of_total']:>7.1f}%")
    
    # Detailed breakdown for top 5 zones
    print("\n" + "=" * 120)
    print("TOP 5 ZONES - SPECIES BREAKDOWN")
    print("=" * 120)
    
    for i, zone in enumerate(zones[:5]):
        print(f"\n🏆 ZONE #{i+1}: {zone['center_lat']:.3f}°, {zone['center_lon']:.3f}°")
        print(f"   📍 Location: {get_region_name(zone['center_lat'], zone['center_lon'])}")
        print(f"   🦈 Total sharks: {zone['total_sharks']:,} ({zone['percentage_of_total']:.1f}% of all data)")
        print(f"   📏 Area: {zone['area_km2']:,.0f} km² ({abs(zone['lat_range'][1]-zone['lat_range'][0]):.2f}° × {abs(zone['lon_range'][1]-zone['lon_range'][0]):.2f}°)")
        print(f"   📊 Density: {zone['density_per_km2']:.3f} sharks/km²")
        print(f"   🗓️  Time span: {zone['year_range']}")
        print(f"   🐟 Species diversity: {zone['species_count']} species")
        
        print(f"   Species breakdown:")
        for species, count in zone['species_breakdown'].items():
            pct = 100 * count / zone['total_sharks']
            print(f"     • {species:40s}: {count:6,} ({pct:5.1f}%)")
    
    # Save detailed results
    zones_df = pd.DataFrame(zones[:10])
    zones_df.to_csv('outputs/top_10_shark_zones.csv', index=False)
    
    # Create visualization
    create_zone_visualization(zones[:10])
    
    print(f"\n✅ Analysis complete!")
    print(f"   📊 Detailed results saved: outputs/top_10_shark_zones.csv")
    print(f"   📈 Visualization saved: outputs/top_10_zones_analysis.png")
    
    return zones[:10]

def get_region_name(lat, lon):
    """Get approximate region name from coordinates"""
    if 35 <= lat <= 55 and -80 <= lon <= -50:
        return "North Atlantic (Nova Scotia/Maritime Canada)"
    elif -40 <= lat <= -25 and 15 <= lon <= 35:
        return "South Africa (Cape Region)"
    elif -45 <= lat <= -30 and 110 <= lon <= 180:
        return "Australia/New Zealand"
    elif 20 <= lat <= 50 and -130 <= lon <= -100:
        return "North Pacific (US West Coast)"
    elif -20 <= lat <= 20 and -180 <= lon <= 180:
        return "Tropical/Equatorial"
    else:
        return f"Unknown Region ({lat:.1f}°, {lon:.1f}°)"

def create_zone_visualization(zones):
    """Create visualization of top zones"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Zone sizes (bar chart)
    zone_ids = [f"Zone {z['zone_id']}" for z in zones]
    shark_counts = [z['total_sharks'] for z in zones]
    
    bars = ax1.bar(zone_ids, shark_counts, color='steelblue', alpha=0.7)
    ax1.set_title('Shark Count by Zone', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Sharks')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, shark_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Density scatter plot
    areas = [z['area_km2'] for z in zones]
    densities = [z['density_per_km2'] for z in zones]
    
    scatter = ax2.scatter(areas, densities, s=[z['total_sharks']/10 for z in zones],
                         c=range(len(zones)), cmap='viridis', alpha=0.7)
    ax2.set_title('Zone Density vs Area', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Area (km²)')
    ax2.set_ylabel('Density (sharks/km²)')
    ax2.set_xscale('log')
    
    # Add zone labels
    for i, zone in enumerate(zones):
        ax2.annotate(f"Z{zone['zone_id']}", (areas[i], densities[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. Species diversity
    species_counts = [z['species_count'] for z in zones]
    
    ax3.bar(zone_ids, species_counts, color='coral', alpha=0.7)
    ax3.set_title('Species Diversity by Zone', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Species')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Geographic distribution
    lats = [z['center_lat'] for z in zones]
    lons = [z['center_lon'] for z in zones]
    
    scatter = ax4.scatter(lons, lats, s=[z['total_sharks']/5 for z in zones],
                         c=range(len(zones)), cmap='plasma', alpha=0.7)
    ax4.set_title('Geographic Distribution of Top Zones', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    # Add zone labels
    for i, zone in enumerate(zones):
        ax4.annotate(f"Z{zone['zone_id']}", (lons[i], lats[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/top_10_zones_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    try:
        zones = analyze_top_10_zones()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()