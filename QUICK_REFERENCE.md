# 🦈 QUICK REFERENCE GUIDE

## 🚀 Quick Commands

```bash
# Run complete analysis
python main_shark_analysis.py

# View results + open map
python view_results.py

# Open map directly
start outputs/shark_habitat_interactive_map.html
```

## 📊 What You Get

### Interactive Map
- **File**: `outputs/shark_habitat_interactive_map.html`
- **Size**: 563 KB
- **Layers**: 6 toggleable visualization layers
- **Features**: Click, zoom, pan, explore

### Model Files
- **Trained Model**: `shark_habitat_model.pkl` (2.9 MB)
- **Performance**: 99.8% accuracy, AUC 1.000

### Visualizations
- **Feature Importance**: `feature_importance.png`
- **Model Evaluation**: `model_evaluation.png` (ROC curves)
- **Hotspot Analysis**: `hotspots_foraging_analysis.png`

## 🗺️ Map Layers (Toggle On/Off)

| Layer | Description | Markers |
|-------|-------------|---------|
| 🦈 Shark Observations | Individual sightings | Blue circles |
| 🔥 Hotspot Heatmap | Density visualization | Yellow→Red gradient |
| ⭐ Top Clusters | Numbered hotspots | Red stars (#1-6) |
| 🐟 Foraging Zones | Feeding habitats | Green→Red scale |
| 📊 Suitability Grid | Probability grid | Color-coded cells |
| 🌊 Probability Map | Overall heatmap | Blue→Yellow→Red |

## 📈 Key Results

**Species**: Prionace glauca (Blue Shark)  
**Observations**: 7,432  
**Hotspots**: 6 clusters identified  
**Top Hotspot**: 44.15°N, 63.34°W (7,037 sharks)  
**Region**: Northwest Atlantic (Nova Scotia)

## 🔧 Customization

### Change Species
```python
# In main_shark_analysis.py, line ~60
target_species = 'Carcharodon carcharias'  # Great White
```

### Adjust Hotspot Sensitivity
```python
# More sensitive (more clusters)
eps=0.3, min_samples=5

# Less sensitive (fewer clusters)
eps=1.0, min_samples=20
```

### Change Map Style
```python
# In main_shark_analysis.py, line ~200
viz.create_base_map(tiles='CartoDB positron')  # Clean
# Or: 'Stamen Terrain'     # Topographic
# Or: 'CartoDB dark_matter' # Dark mode
```

## 📁 File Structure

```
sharkxx/
├── main_shark_analysis.py        ⭐ RUN THIS
├── view_results.py                🗺️ OPEN MAP
├── data_preparation.py
├── random_forest_model.py
├── hotspot_analyzer.py
├── map_visualizer.py
├── README.md                      📖 Full docs
├── PROJECT_SUMMARY.md             📊 Overview
└── outputs/
    ├── shark_habitat_interactive_map.html  ⬅️ MAIN OUTPUT
    ├── shark_habitat_model.pkl
    ├── feature_importance.png
    ├── model_evaluation.png
    ├── hotspots_foraging_analysis.png
    └── analysis_summary.csv
```

## ⚡ Quick Tips

1. **Open Map**: Just double-click the HTML file
2. **Toggle Layers**: Use control panel in top-right
3. **Click Markers**: Get popup with details
4. **Zoom**: Mouse wheel or +/- buttons
5. **Pan**: Click and drag

## 🎯 Use Cases

✅ **Find shark hotspots** → Check red star markers  
✅ **Identify foraging zones** → Enable foraging layer  
✅ **Predict presence** → Check probability heatmap  
✅ **Understand patterns** → View feature importance  
✅ **Plan conservation** → Prioritize hotspot areas  

## 📞 Help

- **Documentation**: See `README.md`
- **Methodology**: See `ML_RECOMMENDATION_REPORT.md`
- **Data Info**: See `DATASET_ANALYSIS_REPORT.md`

## ✅ Checklist

- [x] Random Forest trained (99.8% accuracy)
- [x] Hotspots detected (6 clusters)
- [x] Interactive map created (6 layers)
- [x] Probability predictions generated
- [x] Feature importance analyzed
- [x] All visualizations saved

**Status: READY TO USE! 🎉**
