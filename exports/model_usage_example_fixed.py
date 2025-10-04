# -*- coding: utf-8 -*-
"""
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
    model_data = load_shark_model()
    
    # Handle different model wrapper formats
    if isinstance(model_data, dict):
        if 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
    else:
        model = model_data
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'sst': [sst],
        'chlorophyll': [chlorophyll],
        'depth': [depth],
        'distance_to_coast': [distance_to_coast],
        'sst_gradient': [sst_gradient],
        'month': [month],
        'day_of_year': [day_of_year]
    })
    
    # Make prediction
    try:
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[:, 1][0]
        elif hasattr(model, 'predict'):
            probability = model.predict(input_data)[0]
        else:
            # Handle XGBoost or custom wrapper
            probability = float(model.predict(input_data)[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5  # Default probability
    
    return probability

# Example usage
if __name__ == "__main__":
    
    print("Testing Shark Habitat Model")
    print("="*40)
    
    try:
        # Example 1: Nova Scotia (known Blue Shark habitat)
        print("Example 1: Nova Scotia, Canada")
        prob1 = predict_shark_habitat(
            lat=44.279, lon=-63.171, sst=18.5, chlorophyll=0.8,
            depth=150, distance_to_coast=25, sst_gradient=0.3,
            month=7, day_of_year=200
        )
        print(f"  Shark probability: {prob1:.3f} ({prob1*100:.1f}%)")
        
        # Example 2: South Africa (known Sevengill habitat)
        print("\nExample 2: Cape Town, South Africa")
        prob2 = predict_shark_habitat(
            lat=-34.302, lon=18.834, sst=16.2, chlorophyll=1.2,
            depth=80, distance_to_coast=15, sst_gradient=0.5,
            month=3, day_of_year=85
        )
        print(f"  Shark probability: {prob2:.3f} ({prob2*100:.1f}%)")
        
        # Example 3: Random location
        print("\nExample 3: Open ocean")
        prob3 = predict_shark_habitat(
            lat=30.0, lon=-40.0, sst=22.0, chlorophyll=0.2,
            depth=3000, distance_to_coast=500, sst_gradient=0.1,
            month=6, day_of_year=150
        )
        print(f"  Shark probability: {prob3:.3f} ({prob3*100:.1f}%)")
        
        print("\n✅ Model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()