# 🦈 Shark Habitat Modeling System
### Random Forest Machine Learning + OSM Maps + Hotspot Analysis

Complete end-to-end system for analyzing shark occurrence data, identifying foraging habitats, detecting hotspots, and visualizing results on interactive OpenStreetMap layers.

---

## ✨ Features

✅ **Random Forest Classifier** - State-of-the-art ML for habitat suitability  
✅ **Hotspot Detection** - KDE and clustering methods to find shark aggregations  
✅ **Foraging Habitat Identification** - Environmental analysis for feeding zones  
✅ **Interactive OSM Maps** - Beautiful web maps with multiple layers  
✅ **Probability Predictions** - Predict shark presence at any location  
✅ **Feature Importance Analysis** - Understand which factors matter most  

---

## 🚀 Quick Start

### Run the Complete Analysis

```bash
python main_shark_analysis.py
```

That's it! The system will:
1. Load shark occurrence data from `spottings/` folder
2. Train a Random Forest model
3. Identify hotspots and foraging habitats
4. Generate interactive maps
5. Save all outputs to `outputs/` folder

### View Results

Open `outputs/shark_habitat_interactive_map.html` in your web browser to explore the interactive map!

---

## 📊 What You Get

### Model Performance
- **ROC-AUC Score: 1.000** (Perfect discrimination!)
- **Cross-validation accuracy: 99.8%**
- **Feature importance rankings** showing which environmental factors drive shark presence

### Hotspot Analysis
For **Prionace glauca (Blue Shark)**:
- **6 distinct hotspot clusters** identified
- **Largest hotspot**: 7,037 shark observations (Northwest Atlantic)
- Geographic coordinates and sizes for all clusters

### Interactive Map Layers

The HTML map includes toggleable layers:

1. **🦈 Shark Observations** - Individual sighting points
2. **🔥 Hotspot Heatmap** - Kernel density estimation showing aggregation zones
3. **⭐ Top Hotspot Clusters** - Numbered markers for top 10 hotspots
4. **🐟 Foraging Habitats** - Areas with optimal feeding conditions
5. **📊 Habitat Suitability Grid** - Probability predictions across region
6. **🌡️ Shark Presence Probability** - Heatmap of likelihood

---

## 📁 Project Structure

```
d:\sharkxx\
│
├── main_shark_analysis.py          # ⭐ MAIN SCRIPT - Run this!
│
├── data_preparation.py              # Data loading and feature extraction
├── random_forest_model.py           # Random Forest classifier
├── hotspot_analyzer.py              # Hotspot detection and foraging analysis
├── map_visualizer.py                # Interactive OSM map creation
│
├── spottings/                       # Input data folder
│   ├── 0046525-250920141307145/
│   ├── 0046528-250920141307145/
│   └── 0046539-250920141307145/
│
└── outputs/                         # Results folder (auto-created)
    ├── shark_habitat_interactive_map.html    ← Open this!
    ├── shark_habitat_model.pkl              # Trained model
    ├── feature_importance.png               # Feature rankings
    ├── model_evaluation.png                 # ROC curves
    ├── hotspots_foraging_analysis.png       # Spatial analysis
    └── analysis_summary.csv                 # Summary statistics
```

---

## 🔬 Model Details

### Random Forest Configuration

```python
n_estimators = 500        # Number of decision trees
max_depth = 20            # Maximum tree depth
min_samples_split = 10    # Minimum samples to split node
class_weight = 'balanced' # Handle class imbalance
```

### Features Used (9 total)

**Spatial:**
- Latitude, Longitude
- Distance to coast

**Environmental:**
- Sea Surface Temperature (SST)
- Chlorophyll-a concentration
- Bathymetric depth
- SST gradient (proxy for fronts)

**Temporal:**
- Month
- Day of year

### Feature Importance Rankings

```
1. Latitude           34.3%  ⭐⭐⭐⭐⭐ (Most important!)
2. Longitude          26.5%  ⭐⭐⭐⭐⭐
3. Distance to Coast  17.5%  ⭐⭐⭐⭐
4. Depth              11.3%  ⭐⭐⭐
5. SST                 7.2%  ⭐⭐
6. Day of Year         2.4%  ⭐
7. Month               0.7%  
8. Chlorophyll         0.05%
9. SST Gradient        0.02%
```

**Key Insight:** Location (lat/lon) is the strongest predictor, followed by distance to coast and depth. This suggests sharks have strong spatial preferences.

---

## 🗺️ Hotspot Detection Methods

### Method 1: Kernel Density Estimation (KDE)
- **Bandwidth**: 0.3° (~33 km)
- **Threshold**: 90th percentile
- **Result**: Smooth probability surface showing density

### Method 2: DBSCAN Clustering
- **Epsilon**: 0.5° (~56 km max distance)
- **Min samples**: 10 points per cluster
- **Result**: 6 distinct clusters identified

### Top 5 Hotspots (Blue Shark - Prionace glauca)

| Rank | Location | Observations | Region |
|------|----------|--------------|--------|
| #1 | 44.15°N, 63.34°W | 7,037 | Northwest Atlantic (Nova Scotia) |
| #2 | 47.36°N, 59.92°W | 219 | Gulf of St. Lawrence |
| #3 | 46.11°N, 59.01°W | 77 | Cabot Strait |
| #4 | 44.62°N, 61.98°W | 30 | Scotian Shelf |
| #5 | 45.16°N, 61.40°W | 11 | Eastern Scotian Shelf |

---

## 🐟 Foraging Habitat Criteria

Optimal foraging zones identified based on:

1. **Sea Surface Temperature**: 15-22°C
   - Optimal for prey species (fish, squid)
   
2. **Chlorophyll-a**: >0.5 mg/m³
   - Indicates high primary productivity
   - More plankton → more fish → more sharks

3. **Foraging Score**: 0-1 scale
   - Combines SST optimality + productivity
   - Higher score = better foraging conditions

---

## 📈 Model Performance Metrics

### Cross-Validation Results (5-fold)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 99.8% ± 0.1% | Excellent |
| **ROC-AUC** | 1.000 ± 0.000 | Perfect discrimination |
| **Precision** | 99.8% ± 0.0% | Very few false positives |
| **Recall** | 99.8% ± 0.1% | Catches nearly all presences |
| **F1-Score** | 99.8% ± 0.1% | Balanced performance |

### Confusion Matrix

```
                Predicted
              Absence  Presence
Actual    
  Absence    10,280       16
  Presence        1    7,431
```

**Interpretation:**
- Only 16 false positives (predicted presence but actually absence)
- Only 1 false negative (missed 1 shark presence)
- **99.9% training accuracy**

⚠️ **Note:** High performance may indicate simulated data. With real satellite data, expect 75-90% accuracy.

---

## 🎯 How to Use the Interactive Map

### Opening the Map

1. Navigate to `outputs/` folder
2. Double-click `shark_habitat_interactive_map.html`
3. Opens in your default web browser

### Map Controls

**Toggle Layers** (top right):
- ☑ Check/uncheck to show/hide layers
- Layer Control panel lets you customize view

**Zoom & Pan:**
- Mouse wheel to zoom in/out
- Click and drag to pan
- Double-click to zoom in

**Click Markers:**
- Click any point/marker for popup details
- Shows coordinates, species, scores

**Map Tiles:**
- Currently using OpenStreetMap
- Can switch to Terrain/Satellite in code

### Layer Descriptions

| Layer | Description | Color Scheme |
|-------|-------------|--------------|
| Shark Observations | Individual sightings | Blue circles |
| Hotspot Heatmap | Density visualization | Yellow → Red |
| Top Hotspot Clusters | Numbered aggregation zones | Red stars |
| Foraging Habitats | Feeding zone suitability | Green (good) to Red (poor) |
| Habitat Suitability Grid | Prediction grid | Green (high) to Red (low) |
| Shark Presence Probability | Overall likelihood | Blue → Yellow → Red |

---

## 🔧 Customization

### Analyzing Different Species

Edit `main_shark_analysis.py` to change target species:

```python
# Line ~60, change from automatic to specific:
target_species = 'Carcharodon carcharias'  # Great White Shark
# Or keep as: target_species = prep.species_list[0]  # Most common
```

### Adjusting Hotspot Sensitivity

**More sensitive (more clusters):**
```python
# hotspot_analyzer.py, line ~150
cluster_hotspots = hotspot_analyzer.detect_hotspots_clustering(
    eps=0.3,          # Smaller distance (was 0.5)
    min_samples=5     # Fewer points needed (was 10)
)
```

**Less sensitive (fewer, larger clusters):**
```python
cluster_hotspots = hotspot_analyzer.detect_hotspots_clustering(
    eps=1.0,          # Larger distance
    min_samples=20    # More points needed
)
```

### Changing Foraging Criteria

```python
# main_shark_analysis.py, line ~180
foraging_analysis = hotspot_analyzer.identify_foraging_habitats(
    sst_optimal_range=(18, 24),       # Warmer water preference
    chlorophyll_threshold=0.3          # Lower productivity threshold
)
```

### Map Styling

Change basemap tiles in `main_shark_analysis.py`:

```python
viz.create_base_map(tiles='CartoDB positron')  # Clean minimal
# Or: 'Stamen Terrain'    # Terrain view
# Or: 'CartoDB dark_matter'  # Dark theme
```

---

## ⚠️ Important Notes

### Current Limitations

1. **Simulated Environmental Data**
   - SST, chlorophyll, and depth are currently SIMULATED
   - For production use, extract from actual satellite NetCDF files
   - See `data_preparation.py` line 150-200 for replacement code

2. **Temporal Mismatch**
   - Shark data: 2013-2024
   - Satellite files: 2025
   - Need historical satellite data for accurate correlation

3. **No True Absences**
   - Using pseudo-absences (random background points)
   - MaxEnt or presence-background methods may be more appropriate

### Improving the Model

**For Production Use:**

1. **Get Real Environmental Data**
   ```python
   # Replace add_environmental_features_simulated() with:
   # extract_sst_from_netcdf(shark_coords, sst_files)
   # extract_chlorophyll_from_netcdf(shark_coords, chl_files)
   ```

2. **Download Historical Satellite Data**
   - NASA EarthData: https://oceancolor.gsfc.nasa.gov/
   - Match shark observation dates (2013-2024)

3. **Add More Features**
   - Sea surface height (from SWOT data)
   - Ocean currents
   - Moon phase (affects behavior)
   - Prey species distribution

4. **Spatial Cross-Validation**
   - Current: Stratified k-fold
   - Better: Leave-region-out CV
   - Prevents spatial autocorrelation bias

---

## 📚 Technical Documentation

### Dependencies

```
Python 3.11+
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
folium
geopandas
scipy
shapely
imbalanced-learn
```

### Module APIs

#### SharkDataPreparation
```python
prep = SharkDataPreparation()
prep.load_shark_data('spottings')
prep.filter_shark_species(min_samples=50)
X, y, metadata = prep.prepare_feature_matrix(target_species='Prionace glauca')
```

#### SharkHabitatModel
```python
model = SharkHabitatModel(n_estimators=500, max_depth=20)
model.train(X, y)
model.cross_validate(X, y, cv=5)
probabilities = model.predict_probability(X_new)
```

#### HotspotAnalyzer
```python
analyzer = HotspotAnalyzer()
kde_hotspots = analyzer.detect_hotspots_kde(lat, lon)
cluster_hotspots = analyzer.detect_hotspots_clustering(lat, lon)
foraging = analyzer.identify_foraging_habitats(lat, lon, sst, chl)
```

#### SharkMapVisualizer
```python
viz = SharkMapVisualizer(center_lat=-30, center_lon=25)
viz.create_base_map(tiles='OpenStreetMap')
viz.add_shark_observations(lat, lon, species)
viz.save_map('output.html')
```

---

## 🎓 Scientific Basis

### Random Forest for Species Distribution Modeling

**Why Random Forest?**
- Non-parametric (no assumptions about data distribution)
- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance
- Works with presence-only data (via pseudo-absences)

**Citations:**
- Breiman (2001) - Original Random Forest paper
- Cutler et al. (2007) - Random Forest for ecology
- Elith et al. (2008) - Species distribution modeling review

### Hotspot Detection

**KDE Method:**
- Bandwidth selection critical
- Silverman's rule of thumb
- Adaptive for irregular distributions

**DBSCAN Clustering:**
- Density-based spatial clustering
- Finds clusters of arbitrary shape
- Robust to outliers
- No need to specify number of clusters

---

## 🔮 Future Enhancements

Potential improvements:

1. **Real-time Predictions**
   - API endpoint for on-demand predictions
   - Input: lat, lon, date → Output: probability

2. **Ensemble Models**
   - Combine Random Forest + XGBoost + MaxEnt
   - Weighted average predictions

3. **Movement Modeling**
   - Sequential tracking data
   - LSTM for trajectory prediction
   - Migration route identification

4. **Climate Change Scenarios**
   - Future SST projections
   - Habitat shift predictions

5. **Mobile App Integration**
   - Report sightings
   - View nearby hotspots
   - Safety alerts

---

## 🤝 Contributing

To improve this system:

1. Replace simulated data with real satellite extractions
2. Add more environmental features
3. Implement spatial cross-validation
4. Add uncertainty quantification
5. Create time-series animations

---

## 📄 License

This analysis system is provided for research and educational purposes.

Shark occurrence data from GBIF (Global Biodiversity Information Facility).

---

## 📞 Support

Questions or issues? Check:
- `ML_RECOMMENDATION_REPORT.md` - Detailed methodology
- `DATASET_ANALYSIS_REPORT.md` - Data overview
- Code comments in each module

---

**🎉 You're all set! Open the HTML map and explore shark habitats!** 🦈🗺️
