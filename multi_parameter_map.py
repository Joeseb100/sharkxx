"""
Enhanced Multi-Parameter Map Visualizer
Creates separate toggleable layers for each environmental parameter
"""

import folium
from folium import plugins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

class MultiParameterMapVisualizer:
    """Enhanced map visualizer with separate parameter layers"""
    
    def __init__(self, center_lat=0, center_lon=0, zoom_start=2):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start
        self.map = None
        
    def create_base_map(self, tiles='OpenStreetMap'):
        """Create base map with layer control"""
        print("=" * 80)
        print("CREATING MULTI-PARAMETER INTERACTIVE MAP")
        print("=" * 80)
        
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=None
        )
        
        # Add multiple tile options
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(self.map)
        folium.TileLayer('CartoDB positron', name='Light Theme').add_to(self.map)
        folium.TileLayer('CartoDB dark_matter', name='Dark Theme').add_to(self.map)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False
        ).add_to(self.map)
        
        print(f"✓ Base map created (centered at {self.center_lat:.5f}, {self.center_lon:.5f})")
        print(f"  Zoom level: {self.zoom_start}")
        print(f"  Tile options: 4 different base maps")
        
        return self.map
    
    def add_parameter_layer(self, lat, lon, values, parameter_name, 
                           colormap='viridis', min_val=None, max_val=None):
        """Add a separate layer for each parameter"""
        
        print(f"\nAdding {parameter_name} parameter layer...")
        
        # Handle missing values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            print(f"  ⚠️ No valid {parameter_name} data found")
            return
        
        lat_clean = np.array(lat)[valid_mask]
        lon_clean = np.array(lon)[valid_mask]
        values_clean = np.array(values)[valid_mask]
        
        # Set value range
        if min_val is None:
            min_val = values_clean.min()
        if max_val is None:
            max_val = values_clean.max()
        
        # Create color mapping
        norm = Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.get_cmap(colormap)
        
        # Create feature group for this parameter
        feature_group = folium.FeatureGroup(name=f'{parameter_name} Distribution')
        
        # Add points with color coding
        for i in range(len(lat_clean)):
            value = values_clean[i]
            color = cmap(norm(value))
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            
            # Create popup text
            popup_text = f"""
            <b>{parameter_name}</b><br>
            Location: {lat_clean[i]:.3f}°, {lon_clean[i]:.3f}°<br>
            Value: {value:.3f}<br>
            Range: {min_val:.3f} - {max_val:.3f}
            """
            
            folium.CircleMarker(
                location=[lat_clean[i], lon_clean[i]],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color='white',
                weight=1,
                fillColor=color_hex,
                fillOpacity=0.7
            ).add_to(feature_group)
        
        # Add to map
        feature_group.add_to(self.map)
        
        # Create custom legend for this parameter
        self._add_parameter_legend(parameter_name, min_val, max_val, colormap)
        
        print(f"✓ Added {parameter_name} layer with {len(lat_clean)} points")
        print(f"  Value range: {min_val:.3f} to {max_val:.3f}")
        print(f"  Colormap: {colormap}")
    
    def add_sst_layer(self, lat, lon, sst_values):
        """Add Sea Surface Temperature layer"""
        self.add_parameter_layer(
            lat, lon, sst_values, 
            'Sea Surface Temperature (°C)',
            colormap='plasma',
            min_val=10, max_val=30
        )
    
    def add_chlorophyll_layer(self, lat, lon, chl_values):
        """Add Chlorophyll-a layer"""
        self.add_parameter_layer(
            lat, lon, chl_values,
            'Chlorophyll-a (mg/m³)',
            colormap='viridis',
            min_val=0, max_val=2
        )
    
    def add_depth_layer(self, lat, lon, depth_values):
        """Add Bathymetry/Depth layer"""
        self.add_parameter_layer(
            lat, lon, depth_values,
            'Water Depth (m)',
            colormap='Blues_r',
            min_val=0, max_val=5000
        )
    
    def add_distance_to_coast_layer(self, lat, lon, distance_values):
        """Add Distance to Coast layer"""
        self.add_parameter_layer(
            lat, lon, distance_values,
            'Distance to Coast (km)',
            colormap='YlOrRd',
            min_val=0, max_val=500
        )
    
    def add_sst_gradient_layer(self, lat, lon, gradient_values):
        """Add SST Gradient layer"""
        self.add_parameter_layer(
            lat, lon, gradient_values,
            'SST Gradient (°C/km)',
            colormap='hot',
            min_val=0, max_val=2
        )
    
    def add_temporal_layers(self, lat, lon, months, day_of_year):
        """Add temporal parameter layers"""
        
        # Month layer
        self.add_parameter_layer(
            lat, lon, months,
            'Month of Observation',
            colormap='hsv',
            min_val=1, max_val=12
        )
        
        # Day of year layer
        self.add_parameter_layer(
            lat, lon, day_of_year,
            'Day of Year',
            colormap='twilight',
            min_val=1, max_val=365
        )
    
    def add_species_layer(self, lat, lon, species_names):
        """Add species distribution layer with different colors per species"""
        
        print(f"\nAdding Species Distribution layer...")
        
        # Get unique species
        unique_species = list(set(species_names))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'brown']
        
        # Create feature group for species
        feature_group = folium.FeatureGroup(name='Species Distribution')
        
        for i, species in enumerate(unique_species):
            color = colors[i % len(colors)]
            species_mask = np.array(species_names) == species
            
            species_lat = np.array(lat)[species_mask]
            species_lon = np.array(lon)[species_mask]
            
            for j in range(len(species_lat)):
                popup_text = f"""
                <b>Species Distribution</b><br>
                Species: {species}<br>
                Location: {species_lat[j]:.3f}°, {species_lon[j]:.3f}°<br>
                Count: {species_mask.sum()} observations
                """
                
                folium.CircleMarker(
                    location=[species_lat[j], species_lon[j]],
                    radius=4,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='white',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(feature_group)
        
        feature_group.add_to(self.map)
        
        print(f"✓ Added Species layer with {len(unique_species)} species")
        for i, species in enumerate(unique_species):
            count = (np.array(species_names) == species).sum()
            print(f"  {species}: {count} observations ({colors[i % len(colors)]})")
    
    def add_probability_layer(self, grid_lat, grid_lon, probabilities):
        """Add habitat suitability probability layer"""
        
        print(f"\nAdding Habitat Suitability layer...")
        
        # Filter high probability areas
        high_prob_mask = probabilities > 0.3
        if not np.any(high_prob_mask):
            print("  ⚠️ No high probability areas found")
            return
        
        prob_lat = np.array(grid_lat)[high_prob_mask]
        prob_lon = np.array(grid_lon)[high_prob_mask]
        prob_values = probabilities[high_prob_mask]
        
        feature_group = folium.FeatureGroup(name='Habitat Suitability Probability')
        
        # Create heatmap data
        heat_data = [[prob_lat[i], prob_lon[i], float(prob_values[i])] 
                     for i in range(len(prob_lat))]
        
        plugins.HeatMap(
            heat_data,
            name='Probability Heatmap',
            min_opacity=0.3,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red'}
        ).add_to(feature_group)
        
        feature_group.add_to(self.map)
        
        print(f"✓ Added Probability layer with {len(prob_lat)} high-suitability cells")
        print(f"  Probability range: {prob_values.min():.3f} to {prob_values.max():.3f}")
    
    def add_hotspot_markers(self, hotspots):
        """Add hotspot cluster markers"""
        
        print(f"\nAdding Hotspot Clusters layer...")
        
        feature_group = folium.FeatureGroup(name='Shark Hotspots')
        
        for i, hotspot in enumerate(hotspots[:10]):  # Top 10
            popup_text = f"""
            <b>Shark Hotspot #{i+1}</b><br>
            Location: {hotspot['center_lat']:.3f}°, {hotspot['center_lon']:.3f}°<br>
            Sharks: {hotspot['size']} observations<br>
            Rank: #{i+1} largest cluster
            """
            
            # Size marker based on number of sharks
            marker_size = min(20 + (hotspot['size'] / 100), 50)
            
            folium.Marker(
                location=[hotspot['center_lat'], hotspot['center_lon']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(
                    color='red' if i < 3 else 'orange',
                    icon='star',
                    prefix='fa'
                )
            ).add_to(feature_group)
            
            # Add circle to show hotspot extent
            folium.Circle(
                location=[hotspot['center_lat'], hotspot['center_lon']],
                radius=marker_size * 1000,  # Convert to meters
                color='red' if i < 3 else 'orange',
                weight=2,
                fillOpacity=0.1
            ).add_to(feature_group)
        
        feature_group.add_to(self.map)
        
        print(f"✓ Added {min(10, len(hotspots))} hotspot markers")
    
    def _add_parameter_legend(self, parameter_name, min_val, max_val, colormap):
        """Add a legend for the parameter"""
        # This would create a custom legend HTML - simplified for now
        pass
    
    def save_map(self, filename='multi_parameter_shark_map.html'):
        """Save the interactive map"""
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(self.map)
        
        # Add custom CSS for better styling
        css = """
        <style>
        .leaflet-control-layers {
            font-size: 14px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 8px;
            padding: 10px;
        }
        .leaflet-control-layers-overlays label {
            padding: 5px;
            border-radius: 4px;
            margin: 2px 0;
            display: block;
        }
        .leaflet-control-layers-overlays label:hover {
            background: rgba(0, 123, 255, 0.1);
        }
        </style>
        """
        
        self.map.get_root().html.add_child(folium.Element(css))
        
        # Save map
        self.map.save(filename)
        
        print(f"\n✓ Multi-parameter map saved: {filename}")
        print(f"  Total layers: ~15+ toggleable layers")
        print(f"  Parameters: SST, Chlorophyll, Depth, Distance to Coast, SST Gradient")
        print(f"  Temporal: Month, Day of Year")  
        print(f"  Species: Individual species distributions")
        print(f"  Analysis: Hotspots, Probability, Suitability")
        
        return filename

def create_complete_parameter_maps(shark_data, grid_data, hotspots, probabilities):
    """Create comprehensive multi-parameter maps"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MULTI-PARAMETER MAP CREATION" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Calculate map center
    center_lat = shark_data['latitude'].median()
    center_lon = shark_data['longitude'].median()
    
    # Determine zoom level
    lat_range = shark_data['latitude'].max() - shark_data['latitude'].min()
    lon_range = shark_data['longitude'].max() - shark_data['longitude'].min()
    
    if lat_range > 50 or lon_range > 100:
        zoom_level = 3
    elif lat_range > 20 or lon_range > 40:
        zoom_level = 5
    else:
        zoom_level = 7
    
    # Create visualizer
    viz = MultiParameterMapVisualizer(
        center_lat=center_lat, 
        center_lon=center_lon, 
        zoom_start=zoom_level
    )
    
    # Create base map
    viz.create_base_map()
    
    # Add all parameter layers
    viz.add_sst_layer(
        shark_data['latitude'], 
        shark_data['longitude'], 
        shark_data['sst']
    )
    
    viz.add_chlorophyll_layer(
        shark_data['latitude'], 
        shark_data['longitude'], 
        shark_data['chlorophyll']
    )
    
    viz.add_depth_layer(
        shark_data['latitude'], 
        shark_data['longitude'], 
        shark_data['depth']
    )
    
    viz.add_distance_to_coast_layer(
        shark_data['latitude'], 
        shark_data['longitude'], 
        shark_data['distance_to_coast']
    )
    
    viz.add_sst_gradient_layer(
        shark_data['latitude'], 
        shark_data['longitude'], 
        shark_data['sst_gradient']
    )
    
    viz.add_temporal_layers(
        shark_data['latitude'], 
        shark_data['longitude'],
        shark_data['month'],
        shark_data['day_of_year']
    )
    
    if 'species' in shark_data.columns:
        viz.add_species_layer(
            shark_data['latitude'], 
            shark_data['longitude'], 
            shark_data['species']
        )
    
    # Add analysis layers
    if hotspots:
        viz.add_hotspot_markers(hotspots)
    
    if grid_data is not None and probabilities is not None:
        viz.add_probability_layer(
            grid_data['latitude'],
            grid_data['longitude'], 
            probabilities
        )
    
    # Save map
    filename = viz.save_map('outputs/multi_parameter_shark_map.html')
    
    return filename

if __name__ == "__main__":
    print("Multi-Parameter Map Visualizer")
    print("Use this module by importing and calling create_complete_parameter_maps()")