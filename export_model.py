"""
Model Export Utility
Export trained shark habitat models in multiple formats for different use cases
"""

import joblib
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def export_model_comprehensive():
    """Export trained model in multiple formats with metadata"""
    
    print("=" * 80)
    print("EXPORTING TRAINED SHARK HABITAT MODEL")
    print("=" * 80)
    
    # Check available models
    model_files = []
    if os.path.exists('outputs/All_Sharks_habitat_model.pkl'):
        model_files.append('outputs/All_Sharks_habitat_model.pkl')
    if os.path.exists('outputs/shark_habitat_model.pkl'):
        model_files.append('outputs/shark_habitat_model.pkl')
    
    if not model_files:
        print("❌ No trained models found!")
        print("   Run the analysis first: python main_global_analysis.py")
        return
    
    print(f"📦 Found {len(model_files)} trained models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file}")
    
    # Load the most recent model
    model_file = model_files[0]  # Use the first (most comprehensive) model
    
    print(f"\n🔍 Loading model: {model_file}")
    
    try:
        # Load with joblib (sklearn standard)
        model_data = joblib.load(model_file)
        print("✓ Model loaded successfully")
        
        # Extract model information
        if hasattr(model_data, 'model'):
            model = model_data.model
            feature_names = model_data.feature_names
            model_type = model_data.model_type if hasattr(model_data, 'model_type') else 'Unknown'
        else:
            model = model_data
            feature_names = None
            model_type = type(model).__name__
        
        print(f"  Model type: {model_type}")
        if feature_names:
            print(f"  Features: {len(feature_names)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # =========================================================================
    # EXPORT 1: JOBLIB FORMAT (SKLEARN STANDARD)
    # =========================================================================
    print(f"\n📤 Export 1: Joblib format (.pkl)")
    
    joblib_file = 'exports/shark_habitat_model_joblib.pkl'
    joblib.dump(model_data, joblib_file)
    print(f"✓ Saved: {joblib_file}")
    print(f"  Usage: model = joblib.load('{joblib_file}')")
    
    # =========================================================================
    # EXPORT 2: PICKLE FORMAT (PYTHON STANDARD)
    # =========================================================================
    print(f"\n📤 Export 2: Pickle format (.pickle)")
    
    pickle_file = 'exports/shark_habitat_model_pickle.pickle'
    with open(pickle_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Saved: {pickle_file}")
    print(f"  Usage: with open('{pickle_file}', 'rb') as f: model = pickle.load(f)")
    
    # =========================================================================
    # EXPORT 3: MODEL METADATA (JSON)
    # =========================================================================
    print(f"\n📤 Export 3: Model metadata (.json)")
    
    metadata = {
        'model_info': {
            'type': model_type,
            'algorithm': 'XGBoost' if 'XGB' in model_type else 'Random Forest',
            'gpu_accelerated': 'GPU' in model_type,
            'creation_date': pd.Timestamp.now().isoformat(),
            'file_size_mb': round(os.path.getsize(model_file) / (1024*1024), 2)
        },
        'features': {
            'count': len(feature_names) if feature_names else 'Unknown',
            'names': feature_names if feature_names else [],
            'types': 'Environmental + Spatial + Temporal'
        },
        'performance': {},
        'training_data': {
            'species': ['Prionace glauca', 'Notorynchus cepedianus', 'Carcharodon carcharias'],
            'geographic_extent': {
                'lat_min': -34.73,
                'lat_max': 48.85,
                'lon_min': -73.76,
                'lon_max': 32.90
            },
            'temporal_range': '2013-2024',
            'total_observations': 10296
        },
        'prediction_example': {
            'input_format': 'pandas DataFrame with columns: ' + str(feature_names) if feature_names else 'Unknown',
            'output_format': 'probability (0-1) of shark presence'
        }
    }
    
    # Add performance metrics if available
    try:
        if os.path.exists('outputs/analysis_summary.csv'):
            summary = pd.read_csv('outputs/analysis_summary.csv')
            if len(summary) > 0:
                metadata['performance'] = {
                    'roc_auc': float(summary.iloc[0].get('ROC-AUC', 0)),
                    'cv_accuracy': float(summary.iloc[0].get('CV Accuracy', 0)),
                    'model_accuracy': float(summary.iloc[0].get('Model Accuracy', 0))
                }
    except:
        pass
    
    json_file = 'exports/shark_habitat_model_metadata.json'
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: {json_file}")
    print(f"  Contains: Model info, features, performance, usage examples")
    
    # =========================================================================
    # EXPORT 4: FEATURE IMPORTANCE (CSV)
    # =========================================================================
    print(f"\n📤 Export 4: Feature importance (.csv)")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_,
                'importance_percent': model.feature_importances_ * 100
            }).sort_values('importance', ascending=False)
            
            importance_file = 'exports/feature_importance.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"✓ Saved: {importance_file}")
            print(f"  Top features: {', '.join(importance_df.head(3)['feature'].tolist())}")
        else:
            print("⚠️  Feature importance not available for this model type")
    except Exception as e:
        print(f"⚠️  Could not extract feature importance: {e}")
    
    # =========================================================================
    # EXPORT 5: PREDICTION TEMPLATE (CSV)
    # =========================================================================
    print(f"\n📤 Export 5: Prediction template (.csv)")
    
    if feature_names:
        # Create template with example values
        template_data = {
            'latitude': [44.279, -34.302, 40.0],
            'longitude': [-63.171, 18.834, -70.0],
            'sst': [18.5, 16.2, 20.0],
            'chlorophyll': [0.8, 1.2, 0.5],
            'depth': [150, 80, 200],
            'distance_to_coast': [25, 15, 50],
            'sst_gradient': [0.3, 0.5, 0.2],
            'month': [7, 3, 9],
            'day_of_year': [200, 85, 250]
        }
        
        # Only include features that exist in the model
        template_filtered = {}
        for feature in feature_names:
            if feature in template_data:
                template_filtered[feature] = template_data[feature]
            else:
                template_filtered[feature] = [0.0, 0.0, 0.0]  # Default values
        
        template_df = pd.DataFrame(template_filtered)
        template_file = 'exports/prediction_template.csv'
        template_df.to_csv(template_file, index=False)
        print(f"✓ Saved: {template_file}")
        print(f"  Usage: Load this template, modify values, then predict")
    
    # =========================================================================
    # EXPORT 6: PYTHON USAGE SCRIPT
    # =========================================================================
    print(f"\n📤 Export 6: Usage example script (.py)")
    
    usage_script = f'''"""
Shark Habitat Model - Usage Example
Load and use the exported model for predictions
"""

import joblib
import pandas as pd
import numpy as np

def load_shark_model():
    """Load the trained shark habitat model"""
    model = joblib.load('shark_habitat_model_joblib.pkl')
    return model

def predict_shark_habitat(lat, lon, sst, chlorophyll, depth, 
                         distance_to_coast, sst_gradient, month, day_of_year):
    """
    Predict shark habitat suitability
    
    Parameters:
    -----------
    lat : float
        Latitude (-90 to 90)
    lon : float  
        Longitude (-180 to 180)
    sst : float
        Sea surface temperature (°C)
    chlorophyll : float
        Chlorophyll-a concentration (mg/m³)
    depth : float
        Water depth (m)
    distance_to_coast : float
        Distance to coast (km)
    sst_gradient : float
        SST gradient (°C/km)
    month : int
        Month (1-12)
    day_of_year : int
        Day of year (1-365)
    
    Returns:
    --------
    float : Probability of shark presence (0-1)
    """
    
    # Load model
    model = load_shark_model()
    
    # Create input DataFrame
    input_data = pd.DataFrame({{
        'latitude': [lat],
        'longitude': [lon],
        'sst': [sst],
        'chlorophyll': [chlorophyll],
        'depth': [depth],
        'distance_to_coast': [distance_to_coast],
        'sst_gradient': [sst_gradient],
        'month': [month],
        'day_of_year': [day_of_year]
    }})
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_data)[:, 1][0]
    elif hasattr(model, 'predict'):
        probability = model.predict(input_data)[0]
    else:
        # Handle custom model wrapper
        probability = model.predict_probability(input_data)[0]
    
    return probability

# Example usage
if __name__ == "__main__":
    
    # Example 1: Nova Scotia (known Blue Shark habitat)
    print("Example 1: Nova Scotia, Canada")
    prob1 = predict_shark_habitat(
        lat=44.279, lon=-63.171, sst=18.5, chlorophyll=0.8,
        depth=150, distance_to_coast=25, sst_gradient=0.3,
        month=7, day_of_year=200
    )
    print(f"  Shark probability: {{prob1:.3f}} ({{prob1*100:.1f}}%)")
    
    # Example 2: South Africa (known Sevengill habitat)
    print("\\nExample 2: Cape Town, South Africa")
    prob2 = predict_shark_habitat(
        lat=-34.302, lon=18.834, sst=16.2, chlorophyll=1.2,
        depth=80, distance_to_coast=15, sst_gradient=0.5,
        month=3, day_of_year=85
    )
    print(f"  Shark probability: {{prob2:.3f}} ({{prob2*100:.1f}}%)")
    
    # Example 3: Random location
    print("\\nExample 3: Open ocean")
    prob3 = predict_shark_habitat(
        lat=30.0, lon=-40.0, sst=22.0, chlorophyll=0.2,
        depth=3000, distance_to_coast=500, sst_gradient=0.1,
        month=6, day_of_year=150
    )
    print(f"  Shark probability: {{prob3:.3f}} ({{prob3*100:.1f}}%)")
'''
    
    script_file = 'exports/model_usage_example.py'
    with open(script_file, 'w') as f:
        f.write(usage_script)
    print(f"✓ Saved: {script_file}")
    print(f"  Contains: Complete working example of model usage")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 28 + "MODEL EXPORT COMPLETE!" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("📁 Exported files:")
    print("  📦 shark_habitat_model_joblib.pkl     - Joblib format (recommended)")
    print("  🐍 shark_habitat_model_pickle.pickle  - Python pickle format")
    print("  📄 shark_habitat_model_metadata.json  - Model information & specs")
    print("  📊 feature_importance.csv             - Feature rankings")
    print("  📝 prediction_template.csv            - Input data template")
    print("  🔧 model_usage_example.py             - Complete usage example")
    
    print("\n🎯 Usage recommendations:")
    print("  🔥 Production deployment → Use joblib.pkl + metadata.json")
    print("  🐍 Python integration   → Use pickle.pickle + usage_example.py")
    print("  📊 Analysis/Research     → Use feature_importance.csv")
    print("  🧪 Testing predictions  → Use prediction_template.csv")
    
    print("\n📱 Quick start:")
    print("  1. Copy exports/ folder to your target system")
    print("  2. Install: pip install joblib pandas scikit-learn xgboost")
    print("  3. Run: python model_usage_example.py")
    
    # Get file sizes
    total_size = 0
    for file in ['shark_habitat_model_joblib.pkl', 'shark_habitat_model_pickle.pickle']:
        if os.path.exists(f'exports/{file}'):
            size = os.path.getsize(f'exports/{file}')
            total_size += size
            print(f"  📏 {file}: {size/(1024*1024):.1f} MB")
    
    print(f"\n💾 Total export size: {total_size/(1024*1024):.1f} MB")
    
    return 'exports/'

if __name__ == "__main__":
    import os
    
    # Create exports directory
    os.makedirs('exports', exist_ok=True)
    
    try:
        export_path = export_model_comprehensive()
        print(f"\\n🎉 Success! Model exported to: {export_path}")
    except Exception as e:
        print(f"\\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()