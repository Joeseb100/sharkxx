"""
GLOBAL SHARK HABITAT ANALYSIS
Expanded version for worldwide shark occurrence data
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preparation import SharkDataPreparation
from random_forest_model import SharkHabitatModel
from hotspot_analyzer import HotspotAnalyzer
from map_visualizer import SharkMapVisualizer

def main_global_analysis():
    """Main execution pipeline for GLOBAL shark habitat analysis"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "GLOBAL SHARK HABITAT MODELING SYSTEM" + " " * 24 + "║")
    print("║" + " " * 12 + "Random Forest + OSM Maps + Worldwide Coverage" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # =========================================================================
    # STEP 1: DATA PREPARATION (NO GEOGRAPHIC FILTERING)
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 1/6: LOADING GLOBAL SHARK DATA" + " " * 42 + "│")
    print("└" + "─" * 78 + "┘")
    
    prep = SharkDataPreparation()
    prep.load_shark_data('spottings')
    
    # Filter for sharks but DON'T restrict by geography
    prep.filter_shark_species(min_samples=50)
    prep.extract_basic_features()
    prep.add_environmental_features_simulated()
    
    # Get ALL unique species (not just one)
    print(f"\n🌍 GLOBAL ANALYSIS")
    print(f"   Total species: {len(prep.species_list)}")
    print(f"   Species included: {prep.species_list}")
    
    # Analyze each species separately OR all combined
    species_choice = input("\n❓ Analyze (1) All species combined, (2) Specific species, or (3) Each separately? [1/2/3]: ")
    
    if species_choice == '2':
        print("\nAvailable species:")
        for i, sp in enumerate(prep.species_list, 1):
            count = (prep.shark_data['species'] == sp).sum()
            print(f"  {i}. {sp} ({count} observations)")
        
        sp_idx = int(input("\nSelect species number: ")) - 1
        target_species = prep.species_list[sp_idx]
        species_to_analyze = [target_species]
        
    elif species_choice == '3':
        species_to_analyze = prep.species_list
        print(f"\n✓ Will analyze each of {len(species_to_analyze)} species separately")
    else:
        # Combine all species
        species_to_analyze = ['ALL_SPECIES']
        print("\n✓ Combining all species into single analysis")
    
    # =========================================================================
    # PROCESS EACH SPECIES
    # =========================================================================
    
    for idx, species in enumerate(species_to_analyze):
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {species} ({idx+1}/{len(species_to_analyze)})")
        print(f"{'='*80}")
        
        # Prepare data for this species
        if species == 'ALL_SPECIES':
            X, y, metadata = prep.prepare_feature_matrix(target_species=None)  # All species
            species_name = 'All_Sharks'
        else:
            X, y, metadata = prep.prepare_feature_matrix(target_species=species)
            species_name = species.replace(' ', '_')
        
        # Skip if insufficient data
        if (metadata['presence'] == 1).sum() < 50:
            print(f"⚠ Skipping {species} - insufficient data")
            continue
        
        # =====================================================================
        # STEP 2: RANDOM FOREST TRAINING
        # =====================================================================
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ STEP 2/6: RANDOM FOREST TRAINING" + " " * 45 + "│")
        print("└" + "─" * 78 + "┘")
        
        model = SharkHabitatModel(n_estimators=500, max_depth=20)
        model.train(X, y, feature_names=X.columns.tolist())
        
        cv_results = model.cross_validate(X, y, cv=5)
        importance_df = model.get_feature_importance(plot=True, 
                                                     save_path=f'outputs/{species_name}_feature_importance.png')
        eval_results = model.evaluate(X, y, 
                                     save_path=f'outputs/{species_name}_model_evaluation.png')
        model.save_model(f'outputs/{species_name}_habitat_model.pkl')
        
        # =====================================================================
        # STEP 3: GLOBAL HOTSPOT DETECTION
        # =====================================================================
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ STEP 3/6: GLOBAL HOTSPOT DETECTION" + " " * 43 + "│")
        print("└" + "─" + "┘")
        
        presence_data = metadata[metadata['presence'] == 1].copy()
        
        # Calculate geographic extent
        lat_range = presence_data['latitude'].max() - presence_data['latitude'].min()
        lon_range = presence_data['longitude'].max() - presence_data['longitude'].min()
        
        print(f"\nGeographic extent:")
        print(f"  Latitude: {presence_data['latitude'].min():.2f}° to {presence_data['latitude'].max():.2f}° (range: {lat_range:.2f}°)")
        print(f"  Longitude: {presence_data['longitude'].min():.2f}° to {presence_data['longitude'].max():.2f}° (range: {lon_range:.2f}°)")
        
        # Adjust clustering parameters based on geographic spread
        # For global data, use larger eps values
        if lat_range > 50 or lon_range > 100:
            eps_value = 2.0  # Large clusters for global data
            kde_bandwidth = 1.0
            print(f"  Scale: GLOBAL - using large clustering parameters")
        elif lat_range > 20 or lon_range > 40:
            eps_value = 1.0  # Medium clusters for regional data
            kde_bandwidth = 0.5
            print(f"  Scale: REGIONAL - using medium clustering parameters")
        else:
            eps_value = 0.5  # Small clusters for local data
            kde_bandwidth = 0.3
            print(f"  Scale: LOCAL - using small clustering parameters")
        
        hotspot_analyzer = HotspotAnalyzer()
        
        # KDE-based hotspots
        kde_hotspots = hotspot_analyzer.detect_hotspots_kde(
            lat=presence_data['latitude'].values,
            lon=presence_data['longitude'].values,
            bandwidth=kde_bandwidth,
            threshold_percentile=90
        )
        
        # Clustering-based hotspots
        cluster_hotspots = hotspot_analyzer.detect_hotspots_clustering(
            lat=presence_data['latitude'].values,
            lon=presence_data['longitude'].values,
            eps=eps_value,
            min_samples=max(10, int(len(presence_data) * 0.01))  # At least 1% of data
        )
        
        # =====================================================================
        # STEP 4: FORAGING HABITAT IDENTIFICATION
        # =====================================================================
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ STEP 4/6: FORAGING HABITAT IDENTIFICATION" + " " * 36 + "│")
        print("└" + "─" * 78 + "┘")
        
        presence_full = prep.shark_data[prep.shark_data['species'] == species].copy() if species != 'ALL_SPECIES' else prep.shark_data.copy()
        
        foraging_analysis = hotspot_analyzer.identify_foraging_habitats(
            lat=presence_full['latitude'].values,
            lon=presence_full['longitude'].values,
            sst=presence_full['sst'].values,
            chlorophyll=presence_full['chlorophyll'].values,
            sst_optimal_range=(15, 22),
            chlorophyll_threshold=0.5
        )
        
        hotspot_analyzer.plot_hotspots_and_foraging(
            save_path=f'outputs/{species_name}_hotspots_foraging_analysis.png'
        )
        
        # =====================================================================
        # STEP 5: GLOBAL PROBABILITY PREDICTIONS
        # =====================================================================
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ STEP 5/6: GENERATING GLOBAL PREDICTIONS" + " " * 38 + "│")
        print("└" + "─" * 78 + "┘")
        
        # Create prediction grid covering actual data extent
        lat_min, lat_max = metadata['latitude'].min(), metadata['latitude'].max()
        lon_min, lon_max = metadata['longitude'].min(), metadata['longitude'].max()
        
        # Expand by 10% for context
        lat_buffer = (lat_max - lat_min) * 0.1
        lon_buffer = (lon_max - lon_min) * 0.1
        
        # Adaptive grid size based on extent
        if lat_range > 100 or lon_range > 180:
            grid_size = 100  # High resolution for global
        elif lat_range > 50 or lon_range > 100:
            grid_size = 75   # Medium resolution for regional
        else:
            grid_size = 50   # Standard resolution for local
        
        print(f"\nCreating {grid_size}x{grid_size} prediction grid...")
        
        lat_grid = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, grid_size)
        lon_grid = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, grid_size)
        
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
        grid_points = pd.DataFrame({
            'latitude': lat_mesh.ravel(),
            'longitude': lon_mesh.ravel()
        })
        
        # Add environmental features (simulated)
        grid_points['month'] = 1
        grid_points['day_of_year'] = 15
        
        base_sst = 18 - (np.abs(grid_points['latitude']) - 30) * 0.5
        grid_points['sst'] = base_sst + np.random.normal(0, 1, len(grid_points))
        grid_points['sst'] = grid_points['sst'].clip(10, 30)
        
        coastal_distance = np.abs(grid_points['longitude'] - grid_points['longitude'].mean()) / 10
        grid_points['chlorophyll'] = 1.5 * np.exp(-coastal_distance / 5) + np.random.uniform(0.05, 0.2, len(grid_points))
        grid_points['chlorophyll'] = grid_points['chlorophyll'].clip(0.01, 10)
        
        grid_points['depth'] = 50 + coastal_distance * 20 + np.random.uniform(0, 100, len(grid_points))
        grid_points['depth'] = grid_points['depth'].clip(0, 5000)
        
        grid_points['distance_to_coast'] = coastal_distance * 10
        grid_points['sst_gradient'] = np.abs(np.random.normal(0, 0.5, len(grid_points)))
        
        # Predict probabilities
        X_grid = grid_points[X.columns]
        grid_probabilities = model.predict_probability(X_grid)
        
        print(f"✓ Predictions complete!")
        print(f"  Probability range: {grid_probabilities.min():.3f} to {grid_probabilities.max():.3f}")
        print(f"  Mean probability: {grid_probabilities.mean():.3f}")
        print(f"  High suitability (>0.7): {(grid_probabilities > 0.7).sum()} locations")
        
        # =====================================================================
        # STEP 6: CREATE GLOBAL INTERACTIVE MAP
        # =====================================================================
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ STEP 6/6: CREATING GLOBAL INTERACTIVE MAP" + " " * 35 + "│")
        print("└" + "─" * 78 + "┘")
        
        # Calculate map center (use median for robustness to outliers)
        center_lat = presence_data['latitude'].median()
        center_lon = presence_data['longitude'].median()
        
        # Determine appropriate zoom level based on geographic extent
        if lat_range > 100 or lon_range > 180:
            zoom_level = 2  # World view
        elif lat_range > 50 or lon_range > 100:
            zoom_level = 4  # Continental view
        elif lat_range > 20 or lon_range > 40:
            zoom_level = 6  # Regional view
        else:
            zoom_level = 7  # Local view
        
        print(f"\nMap configuration:")
        print(f"  Center: {center_lat:.2f}°, {center_lon:.2f}°")
        print(f"  Zoom level: {zoom_level}")
        
        viz = SharkMapVisualizer(center_lat=center_lat, center_lon=center_lon, zoom_start=zoom_level)
        viz.create_base_map(tiles='OpenStreetMap')
        
        # Determine sampling rate for performance
        n_obs = len(presence_data)
        if n_obs > 2000:
            max_points = 1000
        elif n_obs > 1000:
            max_points = 500
        else:
            max_points = n_obs
        
        viz.add_shark_observations(
            lat=presence_data['latitude'].values,
            lon=presence_data['longitude'].values,
            species=presence_data['species'].values if 'species' in presence_data else None,
            max_points=max_points
        )
        
        viz.add_hotspots_kde(
            lon_grid=kde_hotspots['lon_grid'],
            lat_grid=kde_hotspots['lat_grid'],
            density=kde_hotspots['density'],
            threshold=kde_hotspots['threshold']
        )
        
        viz.add_hotspot_clusters(
            hotspots=cluster_hotspots['hotspots'],
            top_n=min(15, len(cluster_hotspots['hotspots']))  # Show up to 15 hotspots
        )
        
        viz.add_foraging_zones(
            lat=foraging_analysis['lat'].values,
            lon=foraging_analysis['lon'].values,
            foraging_score=foraging_analysis['foraging_score'].values
        )
        
        viz.add_probability_heatmap(
            lat=grid_points['latitude'].values,
            lon=grid_points['longitude'].values,
            probability=grid_probabilities
        )
        
        viz.add_prediction_grid(
            grid_lat=grid_points['latitude'].values,
            grid_lon=grid_points['longitude'].values,
            grid_probability=grid_probabilities,
            threshold=0.6
        )
        
        # Save map with species-specific name
        map_file = viz.save_map(f'outputs/{species_name}_global_habitat_map.html')
        
        # =====================================================================
        # SUMMARY FOR THIS SPECIES
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR: {species}")
        print(f"{'='*80}")
        
        print(f"\n🦈 Observations: {(metadata['presence'] == 1).sum()}")
        print(f"🤖 Model ROC-AUC: {eval_results['roc_auc']:.3f}")
        print(f"🔥 Hotspot Clusters: {cluster_hotspots['n_clusters']}")
        if cluster_hotspots['n_clusters'] > 0:
            print(f"   Largest hotspot: {cluster_hotspots['hotspots'][0]['size']} observations")
        print(f"🗺️  Map saved: {map_file}")
        print(f"📊 Grid resolution: {grid_size}x{grid_size} cells")
        
        # Save species-specific summary
        summary = {
            'species': species,
            'n_observations': int((metadata['presence'] == 1).sum()),
            'model_auc': float(eval_results['roc_auc']),
            'cv_accuracy_mean': float(cv_results['accuracy'].mean()),
            'n_hotspot_clusters': int(cluster_hotspots['n_clusters']),
            'geographic_extent': {
                'lat_min': float(presence_data['latitude'].min()),
                'lat_max': float(presence_data['latitude'].max()),
                'lon_min': float(presence_data['longitude'].min()),
                'lon_max': float(presence_data['longitude'].max())
            },
            'map_file': map_file
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'outputs/{species_name}_summary.csv', index=False)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 24 + "GLOBAL ANALYSIS COMPLETE!" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print(f"✅ Analyzed {len(species_to_analyze)} species/groups")
    print(f"📁 All outputs saved to: outputs/")
    print(f"🗺️  Open HTML maps in browser to explore")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    try:
        main_global_analysis()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
