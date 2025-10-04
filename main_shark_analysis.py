"""
MAIN EXECUTION SCRIPT
Complete Shark Habitat Modeling Pipeline with Random Forest

This script:
1. Loads and prepares shark occurrence data
2. Trains Random Forest classifier
3. Identifies hotspots and foraging habitats
4. Creates interactive OSM maps with all layers
5. Generates probability predictions
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

def main():
    """Main execution pipeline"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "SHARK HABITAT MODELING SYSTEM" + " " * 29 + "║")
    print("║" + " " * 15 + "Random Forest + OSM Maps + Hotspot Analysis" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # =========================================================================
    # STEP 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 1/6: DATA PREPARATION" + " " * 51 + "│")
    print("└" + "─" * 78 + "┘")
    
    prep = SharkDataPreparation()
    prep.load_shark_data('spottings')
    prep.filter_shark_species(min_samples=50)
    prep.extract_basic_features()
    prep.add_environmental_features_simulated()
    
    # Select target species (most common)
    if len(prep.species_list) == 0:
        print("\n❌ ERROR: No shark species found with sufficient samples!")
        return
    
    target_species = prep.species_list[0]  # Most common species
    print(f"\n🎯 Target species: {target_species}")
    
    X, y, metadata = prep.prepare_feature_matrix(target_species=target_species)
    
    # =========================================================================
    # STEP 2: RANDOM FOREST TRAINING
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 2/6: RANDOM FOREST TRAINING" + " " * 45 + "│")
    print("└" + "─" * 78 + "┘")
    
    model = SharkHabitatModel(n_estimators=500, max_depth=20)
    model.train(X, y, feature_names=X.columns.tolist())
    
    # Cross-validation
    cv_results = model.cross_validate(X, y, cv=5)
    
    # Feature importance
    importance_df = model.get_feature_importance(plot=True, save_path='outputs/feature_importance.png')
    
    # Evaluation
    eval_results = model.evaluate(X, y, save_path='outputs/model_evaluation.png')
    
    # Save model
    model.save_model('outputs/shark_habitat_model.pkl')
    
    # =========================================================================
    # STEP 3: HOTSPOT DETECTION
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 3/6: HOTSPOT DETECTION" + " " * 50 + "│")
    print("└" + "─" * 78 + "┘")
    
    # Get presence-only data
    presence_data = metadata[metadata['presence'] == 1].copy()
    
    hotspot_analyzer = HotspotAnalyzer()
    
    # KDE-based hotspots
    kde_hotspots = hotspot_analyzer.detect_hotspots_kde(
        lat=presence_data['latitude'].values,
        lon=presence_data['longitude'].values,
        bandwidth=0.3,
        threshold_percentile=90
    )
    
    # Clustering-based hotspots
    cluster_hotspots = hotspot_analyzer.detect_hotspots_clustering(
        lat=presence_data['latitude'].values,
        lon=presence_data['longitude'].values,
        eps=0.5,
        min_samples=10
    )
    
    # =========================================================================
    # STEP 4: FORAGING HABITAT IDENTIFICATION
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 4/6: FORAGING HABITAT IDENTIFICATION" + " " * 36 + "│")
    print("└" + "─" * 78 + "┘")
    
    # Get environmental data for presence points
    presence_full = prep.shark_data[prep.shark_data['species'] == target_species].copy()
    
    foraging_analysis = hotspot_analyzer.identify_foraging_habitats(
        lat=presence_full['latitude'].values,
        lon=presence_full['longitude'].values,
        sst=presence_full['sst'].values,
        chlorophyll=presence_full['chlorophyll'].values,
        sst_optimal_range=(15, 22),
        chlorophyll_threshold=0.5
    )
    
    # Create visualization
    hotspot_analyzer.plot_hotspots_and_foraging(save_path='outputs/hotspots_foraging_analysis.png')
    
    # =========================================================================
    # STEP 5: PROBABILITY PREDICTIONS ON GRID
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 5/6: GENERATING PROBABILITY PREDICTIONS" + " " * 33 + "│")
    print("└" + "─" * 78 + "┘")
    
    # Create prediction grid
    lat_min, lat_max = metadata['latitude'].min(), metadata['latitude'].max()
    lon_min, lon_max = metadata['longitude'].min(), metadata['longitude'].max()
    
    # Expand by 5%
    lat_buffer = (lat_max - lat_min) * 0.05
    lon_buffer = (lon_max - lon_min) * 0.05
    
    # Create grid (50x50 for performance)
    grid_size = 50
    lat_grid = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, grid_size)
    lon_grid = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, grid_size)
    
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    grid_points = pd.DataFrame({
        'latitude': lat_mesh.ravel(),
        'longitude': lon_mesh.ravel()
    })
    
    # Add environmental features (simulated for demonstration)
    print(f"\nCreating prediction grid ({grid_size}x{grid_size} = {len(grid_points)} points)...")
    
    # Simulate environmental features for grid
    grid_points['month'] = 1  # January
    grid_points['day_of_year'] = 15
    
    base_sst = 18 - (np.abs(grid_points['latitude']) - 30) * 0.5
    grid_points['sst'] = base_sst + np.random.normal(0, 1, len(grid_points))
    grid_points['sst'] = grid_points['sst'].clip(10, 30)
    
    coastal_distance = np.abs(grid_points['longitude'] - 25)
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
    
    # =========================================================================
    # STEP 6: CREATE INTERACTIVE OSM MAP
    # =========================================================================
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ STEP 6/6: CREATING INTERACTIVE OSM MAP" + " " * 39 + "│")
    print("└" + "─" * 78 + "┘")
    
    # Calculate map center
    center_lat = presence_data['latitude'].mean()
    center_lon = presence_data['longitude'].mean()
    
    viz = SharkMapVisualizer(center_lat=center_lat, center_lon=center_lon, zoom_start=7)
    viz.create_base_map(tiles='OpenStreetMap')
    
    # Add layers
    viz.add_shark_observations(
        lat=presence_data['latitude'].values,
        lon=presence_data['longitude'].values,
        species=presence_data['species'].values if 'species' in presence_data else None,
        max_points=500
    )
    
    viz.add_hotspots_kde(
        lon_grid=kde_hotspots['lon_grid'],
        lat_grid=kde_hotspots['lat_grid'],
        density=kde_hotspots['density'],
        threshold=kde_hotspots['threshold']
    )
    
    viz.add_hotspot_clusters(
        hotspots=cluster_hotspots['hotspots'],
        top_n=10
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
    
    # Save map
    map_file = viz.save_map('outputs/shark_habitat_interactive_map.html')
    
    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "ANALYSIS COMPLETE!" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("📊 SUMMARY REPORT")
    print("=" * 80)
    print(f"\n🦈 Species Analyzed: {target_species}")
    print(f"   Total observations: {(metadata['presence'] == 1).sum()}")
    
    print(f"\n🤖 Model Performance:")
    print(f"   ROC-AUC Score: {eval_results['roc_auc']:.3f}")
    print(f"   Cross-validation accuracy: {cv_results['accuracy'].mean():.3f} ± {cv_results['accuracy'].std():.3f}")
    
    print(f"\n🔥 Hotspots Identified:")
    print(f"   Number of clusters: {cluster_hotspots['n_clusters']}")
    print(f"   Top hotspot size: {cluster_hotspots['hotspots'][0]['size']} observations")
    
    print(f"\n🐟 Foraging Habitats:")
    print(f"   Potential foraging locations: {foraging_analysis['foraging_potential'].sum()}")
    print(f"   High-quality foraging zones: {(foraging_analysis['foraging_score'] > 0.7).sum()}")
    
    print(f"\n🗺️  Map Visualization:")
    print(f"   Interactive map: {map_file}")
    print(f"   Prediction grid: {grid_size}x{grid_size} cells")
    print(f"   High suitability areas: {(grid_probabilities > 0.7).sum()}")
    
    print(f"\n📁 Output Files:")
    print(f"   • outputs/shark_habitat_model.pkl")
    print(f"   • outputs/feature_importance.png")
    print(f"   • outputs/model_evaluation.png")
    print(f"   • outputs/hotspots_foraging_analysis.png")
    print(f"   • outputs/shark_habitat_interactive_map.html")
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! Open the HTML map in your browser to explore results.")
    print("=" * 80 + "\n")
    
    # Save summary statistics
    summary = {
        'species': target_species,
        'n_observations': int((metadata['presence'] == 1).sum()),
        'model_auc': float(eval_results['roc_auc']),
        'cv_accuracy_mean': float(cv_results['accuracy'].mean()),
        'n_hotspot_clusters': int(cluster_hotspots['n_clusters']),
        'n_foraging_locations': int(foraging_analysis['foraging_potential'].sum()),
        'high_suitability_cells': int((grid_probabilities > 0.7).sum())
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('outputs/analysis_summary.csv', index=False)
    print("✓ Summary statistics saved: outputs/analysis_summary.csv\n")

if __name__ == "__main__":
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Run main pipeline
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
