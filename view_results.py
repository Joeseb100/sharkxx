"""
Quick Results Viewer
Opens the interactive map and displays summary statistics
"""

import os
import webbrowser
import pandas as pd
from pathlib import Path

def display_results():
    """Display analysis results and open interactive map"""
    
    print("\n" + "=" * 80)
    print(" " * 25 + "🦈 SHARK HABITAT ANALYSIS RESULTS")
    print("=" * 80 + "\n")
    
    # Check if analysis has been run
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        print("❌ No results found!")
        print("   Please run 'python main_shark_analysis.py' first.\n")
        return
    
    # Load summary statistics
    summary_file = outputs_dir / 'analysis_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        
        print("📊 ANALYSIS SUMMARY")
        print("-" * 80)
        print(f"\n🦈 Target Species: {df['species'].values[0]}")
        print(f"   Total Observations: {df['n_observations'].values[0]:,}")
        
        print(f"\n🤖 Model Performance:")
        print(f"   ROC-AUC Score: {df['model_auc'].values[0]:.3f}")
        print(f"   Cross-Validation Accuracy: {df['cv_accuracy_mean'].values[0]:.3f}")
        
        if df['model_auc'].values[0] > 0.9:
            print("   ⭐ EXCELLENT performance!")
        elif df['model_auc'].values[0] > 0.8:
            print("   ✓ GOOD performance")
        else:
            print("   ⚠ Fair performance - consider adding more features")
        
        print(f"\n🔥 Hotspots Detected:")
        print(f"   Number of Clusters: {df['n_hotspot_clusters'].values[0]}")
        
        print(f"\n🐟 Foraging Habitats:")
        print(f"   Potential Foraging Locations: {df['n_foraging_locations'].values[0]}")
        
        print(f"\n🗺️  Habitat Suitability:")
        print(f"   High Suitability Areas: {df['high_suitability_cells'].values[0]}")
        
        print("\n" + "=" * 80)
    
    # List output files
    print("\n📁 OUTPUT FILES:")
    print("-" * 80)
    
    output_files = {
        'Interactive Map': 'shark_habitat_interactive_map.html',
        'Trained Model': 'shark_habitat_model.pkl',
        'Feature Importance': 'feature_importance.png',
        'Model Evaluation': 'model_evaluation.png',
        'Hotspot Analysis': 'hotspots_foraging_analysis.png',
        'Summary Stats': 'analysis_summary.csv'
    }
    
    for name, filename in output_files.items():
        filepath = outputs_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"   ✓ {name:25s} - {filename:40s} ({size:,.1f} KB)")
        else:
            print(f"   ✗ {name:25s} - Not found")
    
    print("\n" + "=" * 80)
    
    # Open interactive map
    map_file = outputs_dir / 'shark_habitat_interactive_map.html'
    if map_file.exists():
        print("\n🗺️  Opening Interactive Map in Browser...")
        print("-" * 80)
        
        # Get absolute path
        abs_path = map_file.absolute()
        
        # Open in default browser
        webbrowser.open(f'file://{abs_path}')
        
        print(f"   ✓ Map opened: {abs_path}")
        print("\n   🎨 Map Features:")
        print("      • Toggle layers in top-right control panel")
        print("      • Click markers/points for detailed information")
        print("      • Zoom with mouse wheel or +/- buttons")
        print("      • Pan by clicking and dragging")
        
        print("\n   📍 Available Layers:")
        print("      1. 🦈 Shark Observations (blue circles)")
        print("      2. 🔥 Hotspot Heatmap (yellow to red gradient)")
        print("      3. ⭐ Top Hotspot Clusters (numbered red stars)")
        print("      4. 🐟 Foraging Habitats (green = good, red = poor)")
        print("      5. 📊 Habitat Suitability Grid (probability predictions)")
        print("      6. 🌊 Shark Presence Probability (overall heatmap)")
    else:
        print("\n❌ Interactive map not found!")
        print("   Run 'python main_shark_analysis.py' to generate it.")
    
    print("\n" + "=" * 80)
    
    # Show visualization files
    print("\n📈 STATIC VISUALIZATIONS:")
    print("-" * 80)
    
    viz_files = [
        ('feature_importance.png', 'Feature Importance Rankings'),
        ('model_evaluation.png', 'ROC Curve & Precision-Recall'),
        ('hotspots_foraging_analysis.png', 'Hotspots & Foraging Zones')
    ]
    
    for filename, description in viz_files:
        filepath = outputs_dir / filename
        if filepath.exists():
            print(f"   ✓ {description:40s} - {filename}")
    
    print("\n" + "=" * 80)
    
    # Instructions
    print("\n💡 NEXT STEPS:")
    print("-" * 80)
    print("""
   1. Explore the interactive map in your browser
   2. Review the static visualization images in outputs/ folder
   3. Check feature_importance.png to see which factors matter most
   4. Examine hotspots_foraging_analysis.png for spatial patterns
   
   To customize analysis:
   • Edit main_shark_analysis.py to change parameters
   • Modify foraging criteria, hotspot sensitivity, etc.
   • See README.md for full documentation
    """)
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        display_results()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
