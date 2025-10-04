"""
Hotspot Detection and Foraging Habitat Identification
Uses clustering and environmental analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class HotspotAnalyzer:
    """Identify shark aggregation hotspots and foraging habitats"""
    
    def __init__(self):
        self.hotspots = None
        self.foraging_zones = None
        
    def detect_hotspots_kde(self, lat, lon, bandwidth=0.5, threshold_percentile=90):
        """
        Detect hotspots using Kernel Density Estimation
        
        Parameters:
        -----------
        lat, lon : array-like
            Shark observation coordinates
        bandwidth : float
            KDE bandwidth (in degrees, ~55km per degree)
        threshold_percentile : float
            Percentile threshold for hotspot definition
        """
        print("=" * 80)
        print("DETECTING SHARK HOTSPOTS (KDE METHOD)")
        print("=" * 80)
        
        # Prepare coordinates
        coords = np.vstack([lon, lat])
        
        print(f"\nAnalyzing {len(lat)} shark observations...")
        print(f"  Latitude range: {lat.min():.2f} to {lat.max():.2f}")
        print(f"  Longitude range: {lon.min():.2f} to {lon.max():.2f}")
        
        # Create KDE
        kde = gaussian_kde(coords, bw_method=bandwidth)
        
        # Create grid for evaluation
        lon_min, lon_max = lon.min() - 1, lon.max() + 1
        lat_min, lat_max = lat.min() - 1, lat.max() + 1
        
        grid_size = 100
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_coords = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
        
        # Evaluate KDE on grid
        print("\nCalculating density distribution...")
        density = kde(grid_coords).reshape(lon_mesh.shape)
        
        # Define hotspots as high-density areas
        threshold = np.percentile(density, threshold_percentile)
        hotspot_mask = density >= threshold
        
        print(f"\n✓ Hotspot detection complete!")
        print(f"  Density threshold (p{threshold_percentile}): {threshold:.6f}")
        print(f"  Hotspot grid cells: {hotspot_mask.sum()} / {hotspot_mask.size}")
        
        # Store results
        self.hotspots_kde = {
            'lon_grid': lon_mesh,
            'lat_grid': lat_mesh,
            'density': density,
            'hotspot_mask': hotspot_mask,
            'threshold': threshold
        }
        
        return self.hotspots_kde
    
    def detect_hotspots_clustering(self, lat, lon, eps=0.5, min_samples=5):
        """
        Detect hotspots using DBSCAN clustering
        
        Parameters:
        -----------
        lat, lon : array-like
            Shark observation coordinates
        eps : float
            Maximum distance between points in a cluster (degrees)
        min_samples : int
            Minimum points to form a cluster
        """
        print("\n" + "=" * 80)
        print("DETECTING SHARK HOTSPOTS (CLUSTERING METHOD)")
        print("=" * 80)
        
        # Prepare data
        coords = np.column_stack([lon, lat])
        
        print(f"\nClustering {len(coords)} observations...")
        print(f"  eps (max distance): {eps}° (~{eps*111:.0f} km)")
        print(f"  min_samples: {min_samples}")
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(coords)
        
        # Analyze clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"\n✓ Clustering complete!")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        # Get cluster centers and sizes
        hotspots = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            
            center_lon = cluster_coords[:, 0].mean()
            center_lat = cluster_coords[:, 1].mean()
            size = cluster_mask.sum()
            
            hotspots.append({
                'cluster_id': cluster_id,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'size': size,
                'points': cluster_coords
            })
        
        # Sort by size
        hotspots = sorted(hotspots, key=lambda x: x['size'], reverse=True)
        
        print(f"\nTop 5 hotspots:")
        for i, hs in enumerate(hotspots[:5]):
            print(f"  #{i+1}: ({hs['center_lat']:.3f}, {hs['center_lon']:.3f}) - {hs['size']} sharks")
        
        self.hotspots_clusters = {
            'labels': labels,
            'hotspots': hotspots,
            'n_clusters': n_clusters
        }
        
        return self.hotspots_clusters
    
    def identify_foraging_habitats(self, lat, lon, sst, chlorophyll, 
                                   sst_optimal_range=(15, 22), 
                                   chlorophyll_threshold=0.5):
        """
        Identify potential foraging habitats based on environmental conditions
        
        Foraging zones typically have:
        - Optimal temperature range for prey
        - High chlorophyll (high primary productivity)
        - Often near fronts or upwelling zones
        """
        print("\n" + "=" * 80)
        print("IDENTIFYING FORAGING HABITATS")
        print("=" * 80)
        
        df = pd.DataFrame({
            'lat': lat,
            'lon': lon,
            'sst': sst,
            'chlorophyll': chlorophyll
        })
        
        print(f"\nForaging habitat criteria:")
        print(f"  Optimal SST: {sst_optimal_range[0]}°C - {sst_optimal_range[1]}°C")
        print(f"  High productivity (Chl-a): > {chlorophyll_threshold} mg/m³")
        
        # Define foraging habitat
        sst_optimal = (df['sst'] >= sst_optimal_range[0]) & (df['sst'] <= sst_optimal_range[1])
        high_productivity = df['chlorophyll'] >= chlorophyll_threshold
        
        df['foraging_potential'] = (sst_optimal & high_productivity).astype(int)
        
        # Calculate foraging score (0-1)
        # Higher score = better foraging conditions
        sst_score = 1 - np.abs(df['sst'] - np.mean(sst_optimal_range)) / 10
        sst_score = sst_score.clip(0, 1)
        
        chl_score = np.minimum(df['chlorophyll'] / 2, 1)
        
        df['foraging_score'] = (sst_score + chl_score) / 2
        
        n_foraging = df['foraging_potential'].sum()
        print(f"\n✓ Foraging habitat analysis complete!")
        print(f"  Potential foraging locations: {n_foraging} / {len(df)} ({n_foraging/len(df)*100:.1f}%)")
        print(f"  Mean foraging score: {df['foraging_score'].mean():.3f}")
        
        # Identify top foraging zones (cluster high-scoring areas)
        foraging_points = df[df['foraging_potential'] == 1]
        
        if len(foraging_points) > 0:
            print(f"\nTop foraging zone characteristics:")
            print(f"  SST range: {foraging_points['sst'].min():.1f}°C - {foraging_points['sst'].max():.1f}°C")
            print(f"  Chlorophyll range: {foraging_points['chlorophyll'].min():.2f} - {foraging_points['chlorophyll'].max():.2f} mg/m³")
        
        self.foraging_zones = df
        return df
    
    def plot_hotspots_and_foraging(self, save_path='hotspots_foraging_analysis.png'):
        """Create visualization of hotspots and foraging zones"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATION")
        print("=" * 80)
        
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: KDE Hotspots
        if hasattr(self, 'hotspots_kde'):
            ax1 = plt.subplot(131)
            kde_data = self.hotspots_kde
            
            im1 = ax1.contourf(kde_data['lon_grid'], kde_data['lat_grid'], 
                              kde_data['density'], levels=20, cmap='YlOrRd')
            ax1.contour(kde_data['lon_grid'], kde_data['lat_grid'], 
                       kde_data['hotspot_mask'], levels=[0.5], colors='red', linewidths=2)
            plt.colorbar(im1, ax=ax1, label='Density')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('Shark Hotspots (KDE)', fontweight='bold')
            ax1.grid(alpha=0.3)
        
        # Plot 2: Clustered Hotspots
        if hasattr(self, 'hotspots_clusters'):
            ax2 = plt.subplot(132)
            clusters = self.hotspots_clusters
            
            # Plot all points
            for hotspot in clusters['hotspots']:
                pts = hotspot['points']
                ax2.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.6)
                # Mark center
                ax2.scatter(hotspot['center_lon'], hotspot['center_lat'], 
                          s=200, marker='*', edgecolors='red', linewidths=2, 
                          c='yellow', zorder=10)
            
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title(f'Shark Hotspots ({clusters["n_clusters"]} clusters)', fontweight='bold')
            ax2.grid(alpha=0.3)
        
        # Plot 3: Foraging Zones
        if self.foraging_zones is not None:
            ax3 = plt.subplot(133)
            df = self.foraging_zones
            
            scatter = ax3.scatter(df['lon'], df['lat'], 
                                c=df['foraging_score'], 
                                cmap='RdYlGn', s=30, alpha=0.6,
                                vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax3, label='Foraging Suitability Score')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('Foraging Habitat Suitability', fontweight='bold')
            ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    print("Hotspot and Foraging Habitat Analysis Module")
