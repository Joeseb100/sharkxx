# 🦈 SharkXX - Shark Habitat Prediction System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Joeseb100/sharkxx)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU-orange)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-99.8%25-brightgreen)](/)

## 🎯 **Project Overview**

A comprehensive machine learning system for predicting shark habitat suitability using satellite environmental data and GPU-accelerated XGBoost modeling. This system achieves 99.8% accuracy in habitat prediction using real-world satellite data.

## 📊 **Key Features**

- 🤖 **GPU-Accelerated XGBoost Model** - 99.8% cross-validation accuracy
- 🌍 **Global Coverage** - Analysis of 10,335+ shark observations worldwide
- 🛰️ **Real Satellite Data** - NetCDF processing for SST, chlorophyll, bathymetry
- 🗺️ **Interactive Visualizations** - Multi-parameter maps with species analysis
- 🎯 **Top Zone Detection** - Identifies optimal shark habitat zones
- 📦 **Model Export** - Multiple formats (joblib, pickle, JSON metadata)
- 🔍 **Hotspot Analysis** - Automated detection of shark aggregation areas

## 🚀 **Quick Start**

### Prerequisites
```bash
Python 3.11+
CUDA-capable GPU (for XGBoost acceleration)
```

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/Joeseb100/sharkxx.git
cd sharkxx

# Install dependencies
pip install -r requirements.txt

# Run global analysis
python main_global_analysis.py

# Generate interactive maps
python create_parameter_maps.py

# Export trained model
python export_model.py
```

## 📁 **Repository Structure**

### 🔬 **Core Analysis**
- `main_global_analysis.py` - Primary analysis with GPU acceleration
- `analyze_datasets.py` - Dataset exploration and statistics
- `data_preparation.py` - Data preprocessing utilities
- `random_forest_model.py` - Alternative model implementation

### 🗺️ **Visualization**
- `multi_parameter_map.py` - Multi-parameter interactive mapping
- `create_parameter_maps.py` - Automated map generation
- `map_visualizer.py` - Visualization utilities
- `hotspot_analyzer.py` - Hotspot detection and analysis
- `top_10_zones_analysis.py` - Top zones identification

### 📦 **Model & Export**
- `export_model.py` - Model export in multiple formats
- `exports/` - Trained models and usage examples
- `outputs/` - Analysis results and visualizations

### 📚 **Documentation**
- `README.md` - This file
- `PROJECT_SUMMARY.md` - Detailed project overview
- `NEXT_STEPS_GUIDE.md` - Future development roadmap
- `QUICK_REFERENCE.md` - Usage quick reference

## 🧠 **Model Performance**

```
🎯 Model Type: XGBoost GPU Classifier
📊 Accuracy: 99.8% (Cross-validation)
🔄 ROC-AUC: 1.000
🎛️ Features: 8 environmental parameters
🌍 Coverage: Global (83.6° latitude span)
📈 Training Data: 10,335 shark observations
```

## 🔧 **Key Scripts**

### Run Complete Analysis
```bash
python main_global_analysis.py
```
**Outputs:** 
- Trained XGBoost model
- Interactive habitat map
- Performance metrics
- Feature importance analysis

### Generate Multi-Parameter Maps
```bash
python create_parameter_maps.py
```
**Outputs:**
- Interactive HTML map with toggleable layers
- Parameter summary CSV
- Hotspot detection results

### Export Trained Model
```bash
python export_model.py
```
**Outputs:**
- Joblib format (.pkl)
- Pickle format (.pickle)
- JSON metadata
- Usage examples

## 📊 **Environmental Parameters**

The model uses 8 key environmental features:

1. **Sea Surface Temperature (SST)** - From MODIS/VIIRS satellites
2. **Chlorophyll-a Concentration** - Ocean productivity indicator
3. **Bathymetry** - Ocean depth data
4. **Distance to Shore** - Coastal proximity
5. **Latitude/Longitude** - Geographic coordinates
6. **Month** - Seasonal patterns
7. **Season** - Categorical seasonal data

## 🔬 **Species Coverage**

- **Great White Shark** (*Carcharodon carcharias*)
- **Blue Shark** (*Prionace glauca*)
- **Sevengill Shark** (*Notorynchus cepedianus*)
- **And more...** (5 species total)

## 📈 **Results & Outputs**

### Interactive Maps
- `outputs/All_Sharks_global_habitat_map.html` - Global habitat visualization
- `outputs/multi_parameter_shark_map.html` - Multi-parameter analysis

### Model Files
- `exports/shark_habitat_model_joblib.pkl` - Production-ready model
- `exports/model_metadata.json` - Model configuration and metrics

### Analysis Reports
- `outputs/analysis_summary.csv` - Performance metrics
- `outputs/top_10_shark_zones.csv` - Optimal habitat zones

## 🎯 **Usage Examples**

### Load and Use Trained Model
```python
import joblib
import numpy as np

# Load model
model = joblib.load('exports/shark_habitat_model_joblib.pkl')

# Make prediction (example coordinates)
features = [20.5, 0.8, 1500, 25000, 35.0, -75.0, 6, 2]  # SST, Chl, Depth, etc.
probability = model.predict_proba([features])[0][1]
print(f"Shark habitat probability: {probability:.3f}")
```

### Analyze New Location
```python
from exports.model_usage_example_fixed import predict_shark_habitat

# Predict for Nova Scotia waters
result = predict_shark_habitat(44.0, -64.0, "Nova Scotia Coast")
print(f"Habitat suitability: {result['probability']:.1%}")
```

## 🚀 **Next Steps**

See `NEXT_STEPS_GUIDE.md` for detailed future development options:

- **Real-time data integration** - Live satellite feeds
- **Species-specific modeling** - Individual species models
- **Climate change analysis** - Future habitat predictions
- **Web dashboard deployment** - Interactive prediction tool
- **Mobile application** - Field research companion

## 📄 **Data Sources**

**Note:** Dataset files are excluded from this repository due to size constraints. The analysis uses:

- **Satellite Data:** MODIS/VIIRS SST and Chlorophyll NetCDF files
- **Bathymetry:** SWOT and bathymetric datasets
- **Shark Observations:** GBIF occurrence data
- **Wave Data:** SWOT L2 SSH products

*Contact repository owner for access to datasets.*

## 🔧 **System Requirements**

- **Python:** 3.11+
- **GPU:** CUDA-compatible (recommended for XGBoost acceleration)
- **RAM:** 8GB+ (16GB recommended for large datasets)
- **Storage:** 5GB+ for full analysis outputs

## 📝 **License**

Open source research project. See repository for specific licensing terms.

## 🤝 **Contributing**

Contributions welcome! Areas of interest:
- Additional species data
- Model improvements
- Visualization enhancements
- Documentation updates

## 📧 **Contact**

- **Repository:** [Joeseb100/sharkxx](https://github.com/Joeseb100/sharkxx)
- **Issues:** Use GitHub Issues for bug reports and feature requests

---

**🦈 Building better understanding of shark habitats through data science and machine learning.**