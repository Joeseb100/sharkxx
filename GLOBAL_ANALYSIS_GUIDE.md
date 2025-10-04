# 🌍 GLOBAL SHARK HABITAT ANALYSIS GUIDE

## Overview
This guide explains how to run shark habitat analysis for the **entire world** instead of just North America.

---

## 🚀 Quick Start

### Run Global Analysis
```powershell
python main_global_analysis.py
```

That's it! The new script will automatically:
- ✅ Load ALL shark occurrence data (no geographic filtering)
- ✅ Detect geographic extent and adjust parameters automatically
- ✅ Create appropriately-sized prediction grids
- ✅ Set map zoom levels based on data distribution
- ✅ Generate species-specific or combined analyses

---

## 🆕 What Changed from Regional to Global?

### Key Differences

| Feature | Regional (North America) | Global (Worldwide) |
|---------|-------------------------|-------------------|
| **Geographic Filter** | Restricted to North America bounds | No restrictions - uses ALL data |
| **Clustering (eps)** | Fixed at 0.5° | Adaptive: 0.5° (local) to 2.0° (global) |
| **KDE Bandwidth** | Fixed at 0.3° | Adaptive: 0.3° (local) to 1.0° (global) |
| **Map Zoom** | Fixed at 4 | Adaptive: 2 (world) to 7 (local) |
| **Grid Resolution** | 50×50 cells | Adaptive: 50 (local) to 100 (global) |
| **Map Center** | North America centroid | Median of actual observations |
| **Species Processing** | Single species | Multi-species support |

---

## 📊 Adaptive Parameter Selection

The global script **automatically adjusts** parameters based on your data:

### Geographic Scale Detection
```python
lat_range = data['latitude'].max() - data['latitude'].min()
lon_range = data['longitude'].max() - data['longitude'].min()

if lat_range > 50° OR lon_range > 100°:
    Scale = GLOBAL
    eps = 2.0°
    kde_bandwidth = 1.0°
    zoom = 2
    grid = 100×100

elif lat_range > 20° OR lon_range > 40°:
    Scale = REGIONAL
    eps = 1.0°
    kde_bandwidth = 0.5°
    zoom = 4
    grid = 75×75

else:
    Scale = LOCAL
    eps = 0.5°
    kde_bandwidth = 0.3°
    zoom = 7
    grid = 50×50
```

### Why This Matters
- **Too small parameters on global data** → Thousands of tiny clusters (noise)
- **Too large parameters on local data** → Missing important spatial patterns
- **Adaptive approach** → Optimal detection regardless of geographic extent

---

## 🗺️ Map Improvements for Global Coverage

### 1. **Dynamic Map Centering**
```python
# Old (North America)
center_lat = 40.0  # Fixed
center_lon = -95.0

# New (Global)
center_lat = data['latitude'].median()  # Data-driven
center_lon = data['longitude'].median()
```

**Why median?** Robust to outliers (e.g., single observation in remote location won't shift entire map)

### 2. **Automatic Zoom Levels**
The script calculates appropriate zoom:
- **Zoom 2**: Entire world view (lat_range > 100° or lon_range > 180°)
- **Zoom 4**: Continental view (lat_range > 50°)
- **Zoom 6**: Regional view (lat_range > 20°)
- **Zoom 7**: Local view (smaller extents)

### 3. **Performance Optimization**
For large datasets, the script samples points for visualization:
```python
if n_observations > 2000:
    max_map_points = 1000  # Show 1000 representative points
elif n_observations > 1000:
    max_map_points = 500
else:
    max_map_points = all  # Show all points
```

This keeps maps fast and responsive even with millions of observations.

---

## 🦈 Multi-Species Analysis

The global script supports three analysis modes:

### Mode 1: All Species Combined
```
❓ Analyze (1) All species combined, (2) Specific species, or (3) Each separately? [1/2/3]: 1

→ Creates single model for ALL shark species together
→ Output: All_Sharks_global_habitat_map.html
```

**Use when:** You want overall shark distribution patterns

### Mode 2: Specific Species
```
❓ Analyze (1) All species combined, (2) Specific species, or (3) Each separately? [1/2/3]: 2

Available species:
  1. Prionace glauca (7432 observations)
  2. Carcharodon carcharias (1823 observations)
  3. Notorynchus cepedianus (541 observations)

Select species number: 1

→ Analyzes only Blue Shark
→ Output: Prionace_glauca_global_habitat_map.html
```

**Use when:** You want species-specific habitat preferences

### Mode 3: Each Species Separately
```
❓ Analyze (1) All species combined, (2) Specific species, or (3) Each separately? [1/2/3]: 3

→ Loops through ALL species
→ Creates separate model for each
→ Outputs:
    - Prionace_glauca_global_habitat_map.html
    - Carcharodon_carcharias_global_habitat_map.html
    - Notorynchus_cepedianus_global_habitat_map.html
```

**Use when:** You want comparative analysis across species

---

## 📁 Output Files (Per Species)

When you run global analysis, you get:

```
outputs/
├── Prionace_glauca_global_habitat_map.html      # Interactive map
├── Prionace_glauca_habitat_model.pkl            # Trained model
├── Prionace_glauca_feature_importance.png       # Feature rankings
├── Prionace_glauca_model_evaluation.png         # ROC/PR curves
├── Prionace_glauca_hotspots_foraging_analysis.png  # Spatial plots
├── Prionace_glauca_summary.csv                  # Summary stats
└── (repeat for each species if Mode 3)
```

---

## 🔧 Advanced Customization

### Change Clustering Sensitivity
Edit `main_global_analysis.py`, lines ~140-155:

```python
# Make clustering MORE sensitive (more/smaller clusters)
if lat_range > 50:
    eps_value = 1.0  # Changed from 2.0
    
# Make clustering LESS sensitive (fewer/larger clusters)
if lat_range > 50:
    eps_value = 3.0  # Changed from 2.0
```

### Change Grid Resolution
Edit lines ~220-230:

```python
# Higher resolution (slower but more detailed)
if lat_range > 100:
    grid_size = 150  # Changed from 100

# Lower resolution (faster but coarser)
if lat_range > 100:
    grid_size = 75   # Changed from 100
```

### Change Map Tiles
Edit line ~370:

```python
# Default
viz.create_base_map(tiles='OpenStreetMap')

# Alternative options
viz.create_base_map(tiles='Stamen Terrain')    # Terrain
viz.create_base_map(tiles='Stamen Toner')      # Black/white
viz.create_base_map(tiles='CartoDB positron')  # Light theme
```

---

## 💡 Tips for Global Analysis

### 1. **Start with All Species Combined**
- Fastest way to get overview
- Good for exploratory analysis
- Then drill down to specific species

### 2. **Check Data Quality First**
Run this to see your data distribution:
```python
python analyze_datasets.py
```

Look for:
- Coordinate coverage (should span multiple continents for "global")
- Species representation (min 50 observations per species)
- Temporal coverage (are all years represented?)

### 3. **Interpret Adaptive Parameters**
The script will print:
```
Geographic extent:
  Latitude: -42.35° to 64.12° (range: 106.47°)
  Longitude: -178.23° to 179.84° (range: 358.07°)
  Scale: GLOBAL - using large clustering parameters
```

If you see "LOCAL" scale but expected global:
- Check if data is truly global or clustered in one region
- Consider running multiple regional analyses instead

### 4. **Handle Large Datasets**
For >100,000 observations:
```python
# Add this near line 360 in main_global_analysis.py
if n_obs > 100000:
    # Sample data for faster processing
    sample_frac = 10000 / n_obs
    presence_data = presence_data.sample(frac=sample_frac, random_state=42)
```

---

## 🌐 Real-World Examples

### Example 1: Truly Global Blue Sharks
```
Data extent: Lat -45° to +60°, Lon -180° to +180°
→ GLOBAL scale detected
→ eps = 2.0°, zoom = 2, grid = 100×100
→ Result: World map with major ocean basin hotspots
```

### Example 2: Southern Hemisphere Great Whites
```
Data extent: Lat -50° to -25°, Lon 110° to 170°
→ REGIONAL scale detected
→ eps = 1.0°, zoom = 4, grid = 75×75
→ Result: Australia/New Zealand focused map
```

### Example 3: California Sevengill Sharks
```
Data extent: Lat 32° to 41°, Lon -124° to -117°
→ LOCAL scale detected
→ eps = 0.5°, zoom = 7, grid = 50×50
→ Result: Detailed California coast map
```

---

## 🔍 Comparison: Regional vs Global Scripts

### When to Use Regional (`main_shark_analysis.py`)
✅ You KNOW your data is North America only  
✅ You want fixed parameters for reproducibility  
✅ You're comparing against previous North America analyses  

### When to Use Global (`main_global_analysis.py`)
✅ Data spans multiple continents/ocean basins  
✅ You want automatic parameter optimization  
✅ You're analyzing multiple species with different ranges  
✅ You don't know geographic extent beforehand  

---

## 🛠️ Troubleshooting

### Issue: Map shows wrong region
**Cause:** Data might have coordinate errors (e.g., lat/lon swapped)

**Fix:** Check data first:
```python
import pandas as pd
df = pd.read_csv('spottings/.../occurrence.txt', sep='\t')
print(df[['decimalLatitude', 'decimalLongitude']].describe())
```

Verify:
- Latitude: -90 to +90
- Longitude: -180 to +180

### Issue: Too many/few clusters
**Cause:** eps parameter not suitable for your data scale

**Fix:** Manually set eps in lines 140-155:
```python
eps_value = 1.5  # Try different values: 0.3, 0.5, 1.0, 2.0, 3.0
```

### Issue: Map loads slowly
**Cause:** Too many points being plotted

**Fix:** Reduce max_points in line 360:
```python
max_points = 500  # Change from 1000
```

### Issue: Prediction grid is coarse
**Cause:** Adaptive grid size too small

**Fix:** Increase grid_size in lines 220-230:
```python
grid_size = 150  # Increase from 100
```

---

## 📚 Next Steps

After running global analysis:

1. **Open HTML maps** in browser to explore hotspots
2. **Review species summaries** in `outputs/*_summary.csv`
3. **Compare feature importance** across species
4. **Identify conservation priorities** from hotspot clustering
5. **Replace simulated data** with real satellite extractions (see `README.md`)

---

## ❓ FAQ

**Q: Can I run both regional and global analyses?**  
A: Yes! They create separate output files. Run both to compare.

**Q: How long does global analysis take?**  
A: Depends on data size:
- <1,000 obs: ~30 seconds
- 1,000-10,000 obs: 1-3 minutes
- 10,000-100,000 obs: 5-15 minutes
- >100,000 obs: Consider sampling (see Tips #4)

**Q: Can I analyze non-shark species?**  
A: Yes! Remove shark filtering in `data_preparation.py` line 47-50.

**Q: What if I only have one species?**  
A: Script still works! Choose mode 1 or 2 (they'll be equivalent).

**Q: How accurate are the probability predictions?**  
A: Limited by simulated environmental data. Replace with real satellite data for production use.

---

## 📧 Support

For issues or questions:
1. Check this guide first
2. Review `README.md` for general setup
3. Check `PROJECT_SUMMARY.md` for technical details
4. Inspect script output messages (they're detailed!)

---

**Happy Global Shark Mapping! 🦈🌍**
