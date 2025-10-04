"""
AI/ML Feasibility Analysis for Shark-Ocean Dataset
Evaluates data sufficiency, classification quality, and recommends algorithms
"""

import pandas as pd
import numpy as np
from glob import glob
import os

print("=" * 80)
print("AI/ML MODEL FEASIBILITY ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DATA QUANTITY ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("1. DATA QUANTITY ASSESSMENT")
print("=" * 80)

# Shark occurrence data
occurrence_files = glob('spottings/*/occurrence.txt')
total_records = 0
species_counts = {}

print("\nShark Occurrence Data:")
for file in occurrence_files:
    df = pd.read_csv(file, sep='\t')
    records = len(df)
    total_records += records
    dataset_name = os.path.basename(os.path.dirname(file))
    print(f"  {dataset_name}: {records} records")
    
    if 'species' in df.columns:
        for species, count in df['species'].value_counts().items():
            species_counts[species] = species_counts.get(species, 0) + count

print(f"\nTotal shark observations: {total_records}")
print(f"Unique species: {len(species_counts)}")
print("\nSpecies distribution:")
for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {species}: {count} ({count/total_records*100:.1f}%)")

# Satellite data
chl_files = len([f for f in glob('chl/*.nc') if '(1)' not in f])
sst_files = len([f for f in glob('sst/*.nc') if '(1)' not in f and 'AQUA' not in f])
swot_files = len([f for f in glob('waves/*.nc') if '(1)' not in f])

print(f"\nSatellite Data Files:")
print(f"  Chlorophyll: {chl_files} daily files (Aug 27-31, 2025)")
print(f"  SST: {sst_files} hourly files (Apr 3, 2025)")
print(f"  SWOT SSH: {swot_files} files (Oct 1, 2025)")

# ============================================================================
# 2. DATA CLASSIFICATION QUALITY
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA CLASSIFICATION QUALITY ASSESSMENT")
print("=" * 80)

# Check first dataset in detail
sample_file = occurrence_files[0]
df = pd.read_csv(sample_file, sep='\t')

print(f"\nAnalyzing: {os.path.basename(os.path.dirname(sample_file))}")
print(f"Total columns: {len(df.columns)}")

# Check key classification fields
print("\n--- Classification Fields ---")
classification_fields = ['species', 'scientificName', 'taxonKey', 'kingdom', 
                         'phylum', 'class', 'order', 'family', 'genus']
for field in classification_fields:
    if field in df.columns:
        non_null = df[field].notna().sum()
        unique = df[field].nunique()
        print(f"  {field}: {non_null}/{len(df)} non-null ({non_null/len(df)*100:.1f}%), {unique} unique values")

# Check environmental/contextual fields
print("\n--- Spatial/Temporal Fields ---")
spatial_temporal = ['decimalLatitude', 'decimalLongitude', 'eventDate', 
                    'coordinateUncertaintyInMeters', 'depth', 'depthAccuracy']
for field in spatial_temporal:
    if field in df.columns:
        non_null = df[field].notna().sum()
        print(f"  {field}: {non_null}/{len(df)} non-null ({non_null/len(df)*100:.1f}%)")

# Check behavioral/biological fields
print("\n--- Biological/Behavioral Fields ---")
bio_fields = ['lifeStage', 'sex', 'individualCount', 'occurrenceStatus', 
              'establishmentMeans', 'basisOfRecord']
for field in bio_fields:
    if field in df.columns:
        non_null = df[field].notna().sum()
        if non_null > 0:
            unique = df[field].nunique()
            print(f"  {field}: {non_null}/{len(df)} non-null ({non_null/len(df)*100:.1f}%), {unique} unique values")

# ============================================================================
# 3. DATA SUFFICIENCY FOR ML
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA SUFFICIENCY FOR MACHINE LEARNING")
print("=" * 80)

print("\n✓ SUFFICIENT FOR:")
print("  - Binary classification (Presence/Absence at locations)")
print("  - Habitat suitability modeling")
print("  - Environmental preference analysis")
print("  - Spatial distribution modeling")

print("\n⚠ LIMITATIONS:")
limitations = [
    ("Class imbalance", f"{max(species_counts.values())/total_records*100:.1f}% dominated by one species"),
    ("Temporal mismatch", "4-year gap between shark data (2013-2021) and satellite data (2025)"),
    ("Limited species diversity", f"Only {len(species_counts)} species in dataset"),
    ("Small sample size", f"{total_records} observations (ideally want 10,000+ for deep learning)"),
    ("Geographic limitation", "Data focused on South African waters only"),
    ("No true negatives", "Only presence data, no confirmed absence locations")
]

for limitation, detail in limitations:
    print(f"  ❌ {limitation}: {detail}")

# ============================================================================
# 4. TEMPORAL MISMATCH ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. TEMPORAL ALIGNMENT ANALYSIS")
print("=" * 80)

# Check shark data temporal range
if 'eventDate' in df.columns:
    dates = pd.to_datetime(df['eventDate'], errors='coerce')
    shark_start = dates.min()
    shark_end = dates.max()
    
    print(f"\nShark observations: {shark_start.date()} to {shark_end.date()}")
    print(f"Satellite data (CHL): August 27-31, 2025")
    print(f"Satellite data (SST): April 3, 2025")
    print(f"Satellite data (SWOT): October 1, 2025")
    
    gap_years = (pd.Timestamp('2025-04-03') - shark_end).days / 365.25
    print(f"\n⚠ CRITICAL ISSUE: {gap_years:.1f} year gap between shark and satellite data!")
    print("   Cannot directly correlate current satellite conditions with historical shark locations")

# ============================================================================
# 5. RECOMMENDED ML APPROACHES
# ============================================================================
print("\n" + "=" * 80)
print("5. RECOMMENDED MACHINE LEARNING ALGORITHMS")
print("=" * 80)

recommendations = """
┌────────────────────────────────────────────────────────────────────────────┐
│ BEST ALGORITHMS FOR YOUR DATA                                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ 🥇 TIER 1: HIGHLY RECOMMENDED                                             │
│                                                                            │
│ 1. RANDOM FOREST CLASSIFIER / REGRESSOR                                   │
│    ✓ Best for: Species presence/absence, habitat suitability             │
│    ✓ Pros: Handles non-linear relationships, robust to outliers          │
│    ✓ Pros: Works well with small datasets (1000-10000 samples)           │
│    ✓ Pros: Provides feature importance rankings                          │
│    ✓ Pros: Minimal hyperparameter tuning needed                          │
│    Use case: Predict shark presence based on SST, chlorophyll, SSH       │
│                                                                            │
│ 2. GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)                     │
│    ✓ Best for: High-accuracy predictions with tabular data               │
│    ✓ Pros: Often achieves best performance on structured data            │
│    ✓ Pros: Handles missing values well                                   │
│    ✓ Pros: Can model complex interactions between features               │
│    Use case: Predict optimal shark habitat zones                          │
│                                                                            │
│ 3. MAXENT (Maximum Entropy Species Distribution Modeling)                 │
│    ✓ Best for: Presence-only data (no absence records)                   │
│    ✓ Pros: Specifically designed for species distribution                │
│    ✓ Pros: Works with limited occurrence data                            │
│    ✓ Pros: Standard in ecological modeling                               │
│    Use case: Model species distribution from presence points only         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ 🥈 TIER 2: GOOD OPTIONS                                                   │
│                                                                            │
│ 4. SUPPORT VECTOR MACHINES (SVM)                                          │
│    ✓ Best for: Binary classification with clear boundaries               │
│    ✓ Pros: Effective in high-dimensional spaces                          │
│    ⚠ Cons: Requires feature scaling, slower on large datasets            │
│    Use case: Species A vs Species B classification                        │
│                                                                            │
│ 5. K-NEAREST NEIGHBORS (KNN)                                              │
│    ✓ Best for: Spatial interpolation, simple baseline model              │
│    ✓ Pros: Simple to implement and interpret                             │
│    ⚠ Cons: Sensitive to feature scaling, slow prediction                 │
│    Use case: Quick baseline for spatial predictions                       │
│                                                                            │
│ 6. GAUSSIAN PROCESSES / KRIGING                                           │
│    ✓ Best for: Spatial interpolation with uncertainty estimates          │
│    ✓ Pros: Provides prediction uncertainty                               │
│    ✓ Pros: Good for geostatistical analysis                              │
│    ⚠ Cons: Computationally expensive for large datasets                  │
│    Use case: Interpolate shark presence probability across space          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ 🥉 TIER 3: ADVANCED (Use with caution given data limitations)            │
│                                                                            │
│ 7. NEURAL NETWORKS (Fully Connected)                                      │
│    ⚠ Requires: 10,000+ samples (you have ~3,000)                         │
│    ⚠ Risk: Overfitting with limited data                                 │
│    ✓ Use only if: You can augment data or use transfer learning          │
│                                                                            │
│ 8. CONVOLUTIONAL NEURAL NETWORKS (CNN)                                    │
│    ⚠ Best for: Image-based analysis of satellite data                    │
│    ⚠ Requires: Large labeled dataset (10,000+ images)                    │
│    ✓ Use only if: Extracting patterns from satellite imagery patches     │
│                                                                            │
│ 9. RECURRENT NEURAL NETWORKS (LSTM/GRU)                                   │
│    ✓ Best for: Time-series prediction of shark movements                 │
│    ⚠ Requires: Sequential tracking data (continuous trajectories)        │
│    ⚠ Risk: Current data has temporal gaps                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
"""
print(recommendations)

# ============================================================================
# 6. SPECIFIC USE CASES & MODELS
# ============================================================================
print("\n" + "=" * 80)
print("6. RECOMMENDED MODELS BY USE CASE")
print("=" * 80)

use_cases = """
📊 USE CASE 1: Habitat Suitability Mapping
   Goal: Identify optimal shark habitats based on environmental conditions
   Algorithm: Random Forest Regressor or MaxEnt
   Features: SST, Chlorophyll, SSH, distance to coast, bathymetry
   Output: Probability map of shark presence (0-1)
   Sample size: ✓ Sufficient (2,864 points)

🎯 USE CASE 2: Species Classification
   Goal: Distinguish between Sevengill vs Great White shark habitats
   Algorithm: Random Forest Classifier or XGBoost
   Features: SST, Chlorophyll, SSH, depth, season
   Output: Binary classification (Species A or B)
   Sample size: ⚠ Imbalanced (2789 vs 75 - need SMOTE/oversampling)

📍 USE CASE 3: Presence/Absence Prediction
   Goal: Predict shark presence at new locations
   Algorithm: MaxEnt or Gradient Boosting
   Features: Environmental variables at location
   Output: Presence probability
   Sample size: ⚠ Only presence data (need pseudo-absences)

🌊 USE CASE 4: Environmental Preference Analysis
   Goal: Understand environmental conditions associated with shark sightings
   Algorithm: Random Forest + Feature Importance Analysis
   Features: All environmental variables
   Output: Feature importance rankings, partial dependence plots
   Sample size: ✓ Sufficient

🗺️ USE CASE 5: Spatial Distribution Modeling
   Goal: Create species distribution maps
   Algorithm: MaxEnt or Spatial GLM
   Features: Lat, Lon, environmental covariates
   Output: Geographic distribution map
   Sample size: ✓ Sufficient for presence-only modeling
"""
print(use_cases)

# ============================================================================
# 7. DATA PREPARATION REQUIREMENTS
# ============================================================================
print("\n" + "=" * 80)
print("7. DATA PREPARATION REQUIREMENTS")
print("=" * 80)

preparation_steps = """
CRITICAL STEPS BEFORE MODELING:

1. ⚠ RESOLVE TEMPORAL MISMATCH
   Option A: Download historical satellite data (2013-2021) matching shark dates
   Option B: Collect recent shark tracking data (2025) matching current satellite
   Option C: Use climatological averages (less accurate but feasible)

2. 📍 EXTRACT ENVIRONMENTAL FEATURES
   - Extract SST at each shark observation lat/lon
   - Extract chlorophyll at each shark observation lat/lon
   - Extract SSH at each shark observation lat/lon
   - Create feature matrix: [lat, lon, sst, chl, ssh, depth, date, ...]

3. ⚖️ ADDRESS CLASS IMBALANCE
   - Sevengill shark: 2789 samples (97.4%)
   - Great White: 75 samples (2.6%)
   Solutions:
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weights in model
   - Stratified sampling
   - Focus on single-species models

4. 🎯 GENERATE PSEUDO-ABSENCES (for presence-only data)
   - Random background points in study region
   - Ratio: 1:1 or 1:10 (presence:absence)
   - Ensure absences are spatially separated from presences

5. 🔧 FEATURE ENGINEERING
   - Temporal: Season, month, year, day of year
   - Spatial: Distance to coast, bathymetry, slope
   - Derived: SST gradient, chlorophyll anomaly
   - Lagged: Previous week's SST, chlorophyll trends

6. 📊 TRAIN/TEST SPLIT
   - Spatial cross-validation (not random split!)
   - Leave-one-region-out cross-validation
   - Prevents spatial autocorrelation bias
"""
print(preparation_steps)

# ============================================================================
# 8. FINAL RECOMMENDATION
# ============================================================================
print("\n" + "=" * 80)
print("8. FINAL RECOMMENDATION")
print("=" * 80)

final_rec = """
🎯 RECOMMENDED APPROACH:

PRIMARY ALGORITHM: Random Forest Classifier
BACKUP ALGORITHM: XGBoost Classifier
SPECIALIZED: MaxEnt for presence-only modeling

RATIONALE:
✓ Random Forest works well with ~3,000 samples
✓ Handles non-linear environmental relationships
✓ Provides interpretable feature importance
✓ Robust to outliers and missing data
✓ Minimal hyperparameter tuning required
✓ Good performance even with class imbalance

IMPLEMENTATION STEPS:
1. Acquire historical satellite data (2013-2021) OR use climatology
2. Extract environmental features at shark locations
3. Generate pseudo-absence points
4. Address class imbalance (SMOTE or separate models per species)
5. Train Random Forest with spatial cross-validation
6. Validate with held-out spatial regions
7. Generate habitat suitability maps
8. Analyze feature importance to understand drivers

EXPECTED PERFORMANCE:
- Accuracy: 70-85% (depends on data quality)
- AUC-ROC: 0.75-0.90 (with proper validation)
- Better for Sevengill (more data) than Great White (limited data)

MINIMUM VIABLE MODEL:
- Use only Sevengill shark data (2,789 samples)
- Extract SST and Chlorophyll at locations
- Train Random Forest binary classifier (presence/absence)
- Validate with spatial k-fold cross-validation
- Expected AUC: 0.70-0.80
"""
print(final_rec)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
