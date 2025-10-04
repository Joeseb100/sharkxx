"""
Interactive Map Visualization with OpenStreetMap Integration
Creates interactive HTML maps with Folium
"""

import folium
from folium import plugins
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

class SharkMapVisualizer:
    """Create interactive maps with OSM basemap"""
    
    def __init__(self, center_lat=-30, center_lon=25, zoom_start=6):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start
        self.map = None
        
    def create_base_map(self, tiles='OpenStreetMap'):
        """
        Create base map with OSM tiles
        
        Tile options:
        - 'OpenStreetMap': Standard OSM
        - 'Stamen Terrain': Terrain view
        - 'Stamen Toner': High contrast
        - 'CartoDB positron': Clean minimal
        - 'CartoDB dark_matter': Dark theme
        """
        print("=" * 80)
        print("CREATING INTERACTIVE MAP")
        print("=" * 80)
        
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=tiles,
            control_scale=True
        )
        
        print(f"✓ Base map created (centered at {self.center_lat}, {self.center_lon})")
        print(f"  Tile layer: {tiles}")
        
        return self.map
    
    def add_shark_observations(self, lat, lon, species=None, max_points=1000):
        """Add shark observation points to map"""
        print("\nAdding shark observation points...")
        
        # Subsample if too many points
        if len(lat) > max_points:
            indices = np.random.choice(len(lat), max_points, replace=False)
            lat = lat[indices]
            lon = lon[indices]
            if species is not None:
                species = species[indices]
            print(f"  Subsampled to {max_points} points for performance")
        
        # Create feature group
        shark_layer = folium.FeatureGroup(name='Shark Observations', show=True)
        
        # Add markers
        for i in range(len(lat)):
            sp_name = species[i] if species is not None else 'Unknown'
            
            folium.CircleMarker(
                location=[lat[i], lon[i]],
                radius=4,
                popup=f"<b>{sp_name}</b><br>Lat: {lat[i]:.3f}<br>Lon: {lon[i]:.3f}",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.6,
                weight=1
            ).add_to(shark_layer)
        
        shark_layer.add_to(self.map)
        print(f"✓ Added {len(lat)} shark observations")
        
    def add_hotspots_kde(self, lon_grid, lat_grid, density, threshold, num_contours=5):
        """Add KDE hotspot contours to map"""
        print("\nAdding KDE hotspot contours...")
        
        hotspot_layer = folium.FeatureGroup(name='Hotspot Density (KDE)', show=True)
        
        # Create contour levels
        levels = np.linspace(threshold, density.max(), num_contours)
        
        # Convert density to heatmap data
        # Note: This is a simplified approach; full contour plotting requires additional processing
        print(f"✓ Hotspot density layer prepared ({num_contours} levels)")
        print(f"  Density range: {density.min():.6f} to {density.max():.6f}")
        
        # Add heatmap as alternative to contours
        heat_data = []
        for i in range(0, len(lat_grid.ravel()), 10):  # Subsample for performance
            lat_val = lat_grid.ravel()[i]
            lon_val = lon_grid.ravel()[i]
            density_val = density.ravel()[i]
            if density_val > threshold:
                heat_data.append([lat_val, lon_val, float(density_val)])
        
        if len(heat_data) > 0:
            plugins.HeatMap(
                heat_data,
                name='Hotspot Heatmap',
                min_opacity=0.3,
                max_zoom=13,
                radius=15,
                blur=20,
                gradient={0.0: 'yellow', 0.5: 'orange', 1.0: 'red'}
            ).add_to(self.map)
            print(f"✓ Added heatmap with {len(heat_data)} high-density points")
    
    def add_hotspot_clusters(self, hotspots, top_n=10):
        """Add clustered hotspots with markers"""
        print("\nAdding hotspot cluster markers...")
        
        cluster_layer = folium.FeatureGroup(name='Top Hotspot Clusters', show=True)
        
        # Add top N hotspots
        for i, hotspot in enumerate(hotspots[:top_n]):
            size = hotspot['size']
            lat = hotspot['center_lat']
            lon = hotspot['center_lon']
            
            # Size of marker based on cluster size
            marker_size = min(30, 10 + size / 10)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_size,
                popup=f"<b>Hotspot #{i+1}</b><br>"
                      f"Observations: {size}<br>"
                      f"Location: ({lat:.3f}, {lon:.3f})",
                color='darkred',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                weight=2
            ).add_to(cluster_layer)
            
            # Add label
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-shadow: 2px 2px 4px #000000;">#{i+1}</div>')
            ).add_to(cluster_layer)
        
        cluster_layer.add_to(self.map)
        print(f"✓ Added {min(top_n, len(hotspots))} hotspot markers")
    
    def add_foraging_zones(self, lat, lon, foraging_score):
        """Add foraging habitat zones"""
        print("\nAdding foraging habitat zones...")
        
        foraging_layer = folium.FeatureGroup(name='Foraging Habitats', show=False)
        
        # Color mapping
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('RdYlGn')
        
        # Add points colored by foraging score
        for i in range(len(lat)):
            score = foraging_score[i]
            if score > 0.5:  # Only show good foraging habitats
                color = cmap(norm(score))
                color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                
                folium.CircleMarker(
                    location=[lat[i], lon[i]],
                    radius=5,
                    popup=f"<b>Foraging Score: {score:.2f}</b><br>"
                          f"Lat: {lat[i]:.3f}<br>Lon: {lon[i]:.3f}",
                    color=color_hex,
                    fill=True,
                    fillColor=color_hex,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(foraging_layer)
        
        foraging_layer.add_to(self.map)
        print(f"✓ Added foraging habitat zones")
    
    def add_probability_heatmap(self, lat, lon, probability):
        """Add probability heatmap showing likelihood of shark presence"""
        print("\nAdding probability prediction heatmap...")
        
        # Prepare heatmap data
        heat_data = [[lat[i], lon[i], float(probability[i])] 
                     for i in range(len(lat)) if probability[i] > 0.3]
        
        if len(heat_data) > 0:
            plugins.HeatMap(
                heat_data,
                name='Shark Presence Probability',
                min_opacity=0.2,
                max_zoom=13,
                radius=20,
                blur=25,
                gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
            ).add_to(self.map)
            print(f"✓ Added probability heatmap with {len(heat_data)} predictions")
        else:
            print("  No high-probability areas to display")
    
    def add_prediction_grid(self, grid_lat, grid_lon, grid_probability, threshold=0.5):
        """Add grid of predictions with color-coded cells"""
        print("\nAdding prediction grid...")
        
        grid_layer = folium.FeatureGroup(name='Habitat Suitability Grid', show=False)
        
        # Color mapping
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('RdYlGn')
        
        count = 0
        for i in range(len(grid_lat)):
            prob = grid_probability[i]
            if prob > threshold:
                color = cmap(norm(prob))
                color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                
                folium.CircleMarker(
                    location=[grid_lat[i], grid_lon[i]],
                    radius=8,
                    popup=f"<b>Habitat Suitability: {prob:.1%}</b><br>"
                          f"Location: ({grid_lat[i]:.3f}, {grid_lon[i]:.3f})",
                    color=color_hex,
                    fill=True,
                    fillColor=color_hex,
                    fillOpacity=0.6,
                    weight=0
                ).add_to(grid_layer)
                count += 1
        
        grid_layer.add_to(self.map)
        print(f"✓ Added {count} grid cells with probability > {threshold}")
    
    def save_map(self, filepath='shark_habitat_map.html'):
        """Save interactive map to HTML file"""
        if self.map is None:
            raise ValueError("Map not created yet! Call create_base_map() first.")
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(self.map)
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h3 style="margin: 0;">🦈 Shark Habitat Analysis Map</h3>
        <p style="margin: 5px 0;"><b>Random Forest Model Predictions</b></p>
        <p style="margin: 5px 0; font-size: 11px;">
        Toggle layers in the control panel →<br>
        Click markers for details
        </p>
        </div>
        '''
        self.map.get_root().html.add_child(folium.Element(title_html))
        
        # Save
        self.map.save(filepath)
        print(f"\n✓ Interactive map saved: {filepath}")
        print(f"  Open this file in a web browser to view the map")
        
        return filepath

if __name__ == "__main__":
    print("Interactive Map Visualization Module")
