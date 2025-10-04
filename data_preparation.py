"""
Data Preparation Module for Shark Habitat Modeling
Handles feature extraction, pseudo-absences, and data cleaning
"""

import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SharkDataPreparation:
    """Prepare shark occurrence data for ML modeling"""
    
    def __init__(self):
        self.shark_data = None
        self.feature_matrix = None
        self.species_list = []
        
    def load_shark_data(self, spottings_dir='spottings'):
        """Load all shark occurrence data"""
        print("=" * 80)
        print("LOADING SHARK OCCURRENCE DATA")
        print("=" * 80)
        
        occurrence_files = glob(f'{spottings_dir}/*/occurrence.txt')
        all_data = []
        
        for file in occurrence_files:
            print(f"\nLoading: {file}")
            df = pd.read_csv(file, sep='\t', low_memory=False)
            all_data.append(df)
            print(f"  Records: {len(df)}")
        
        self.shark_data = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total records loaded: {len(self.shark_data)}")
        
        return self.shark_data
    
    def filter_shark_species(self, min_samples=50):
        """Keep only shark species with sufficient samples"""
        print("\n" + "=" * 80)
        print("FILTERING SHARK SPECIES")
        print("=" * 80)
        
        # List of known shark species (some datasets have plant species mixed in!)
        shark_keywords = ['shark', 'carcharodon', 'prionace', 'notorynchus', 
                         'carcharias', 'sphyrna', 'galeocerdo', 'isurus']
        
        # Filter for actual sharks
        shark_mask = self.shark_data['species'].str.lower().apply(
            lambda x: any(keyword in str(x).lower() for keyword in shark_keywords)
        )
        
        self.shark_data = self.shark_data[shark_mask].copy()
        print(f"✓ Filtered to shark species only: {len(self.shark_data)} records")
        
        # Keep species with enough samples
        species_counts = self.shark_data['species'].value_counts()
        valid_species = species_counts[species_counts >= min_samples].index
        
        self.shark_data = self.shark_data[
            self.shark_data['species'].isin(valid_species)
        ].copy()
        
        self.species_list = list(valid_species)
        
        print(f"\nSpecies with {min_samples}+ samples:")
        for species in self.species_list:
            count = (self.shark_data['species'] == species).sum()
            print(f"  {species}: {count}")
        
        print(f"\n✓ Total shark records: {len(self.shark_data)}")
        return self.shark_data
    
    def extract_basic_features(self):
        """Extract basic features from occurrence data"""
        print("\n" + "=" * 80)
        print("EXTRACTING BASIC FEATURES")
        print("=" * 80)
        
        df = self.shark_data.copy()
        
        # Spatial features
        df['latitude'] = df['decimalLatitude']
        df['longitude'] = df['decimalLongitude']
        
        # Temporal features
        df['date'] = pd.to_datetime(df['eventDate'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = df['month'].apply(self._get_season_southern_hemisphere)
        
        # Coordinate uncertainty
        df['coord_uncertainty_km'] = df['coordinateUncertaintyInMeters'] / 1000
        
        # Create presence flag (always 1 for occurrence data)
        df['presence'] = 1
        
        print(f"✓ Extracted basic features")
        print(f"  Temporal range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Spatial extent: Lat {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
        print(f"                  Lon {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
        
        self.shark_data = df
        return df
    
    def _get_season_southern_hemisphere(self, month):
        """Get season for Southern Hemisphere"""
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'
    
    def add_environmental_features_simulated(self):
        """
        Add simulated environmental features
        NOTE: In production, these should be extracted from actual satellite data
        For now, we'll create realistic simulated values based on location/season
        """
        print("\n" + "=" * 80)
        print("ADDING ENVIRONMENTAL FEATURES (SIMULATED)")
        print("=" * 80)
        print("⚠ WARNING: Using simulated data. Replace with actual satellite extraction!")
        
        df = self.shark_data.copy()
        
        # Simulate SST (sea surface temperature) based on latitude and season
        # Warmer near equator, cooler at higher latitudes
        base_sst = 18 - (np.abs(df['latitude']) - 30) * 0.5
        season_effect = df['season'].map({
            'summer': 3, 'autumn': 0, 'winter': -3, 'spring': 0
        })
        df['sst'] = base_sst + season_effect + np.random.normal(0, 1, len(df))
        df['sst'] = df['sst'].clip(10, 30)  # Realistic ocean temps
        
        # Simulate Chlorophyll-a (higher near coast, lower offshore)
        # Use longitude as proxy for coastal vs offshore (very simplified!)
        coastal_distance = np.abs(df['longitude'] - 25)  # Arbitrary coastal reference
        df['chlorophyll'] = 1.5 * np.exp(-coastal_distance / 5) + np.random.uniform(0.05, 0.2, len(df))
        df['chlorophyll'] = df['chlorophyll'].clip(0.01, 10)
        
        # Simulate bathymetry (depth - positive values)
        df['depth'] = 50 + coastal_distance * 20 + np.random.uniform(0, 100, len(df))
        df['depth'] = df['depth'].clip(0, 5000)
        
        # Distance to coast (km) - simplified
        df['distance_to_coast'] = coastal_distance * 10
        
        # SST gradient (proxy for fronts and upwelling)
        df['sst_gradient'] = np.abs(np.random.normal(0, 0.5, len(df)))
        
        print("✓ Simulated environmental features added:")
        print(f"  SST: {df['sst'].min():.1f}°C to {df['sst'].max():.1f}°C (mean: {df['sst'].mean():.1f}°C)")
        print(f"  Chlorophyll: {df['chlorophyll'].min():.3f} to {df['chlorophyll'].max():.3f} mg/m³")
        print(f"  Depth: {df['depth'].min():.0f}m to {df['depth'].max():.0f}m")
        
        self.shark_data = df
        return df
    
    def generate_pseudo_absences(self, ratio=1.0):
        """
        Generate pseudo-absence points (background sampling)
        Essential for presence-only data modeling
        """
        print("\n" + "=" * 80)
        print("GENERATING PSEUDO-ABSENCE POINTS")
        print("=" * 80)
        
        presences = self.shark_data.copy()
        n_absences = int(len(presences) * ratio)
        
        # Get spatial extent
        lat_min, lat_max = presences['latitude'].min(), presences['latitude'].max()
        lon_min, lon_max = presences['longitude'].min(), presences['longitude'].max()
        
        # Expand extent by 10% to include surrounding areas
        lat_buffer = (lat_max - lat_min) * 0.1
        lon_buffer = (lon_max - lon_min) * 0.1
        
        print(f"Generating {n_absences} pseudo-absence points...")
        print(f"Spatial extent: Lat [{lat_min:.2f}, {lat_max:.2f}], Lon [{lon_min:.2f}, {lon_max:.2f}]")
        
        # Generate random points
        absences = pd.DataFrame({
            'latitude': np.random.uniform(lat_min - lat_buffer, lat_max + lat_buffer, n_absences),
            'longitude': np.random.uniform(lon_min - lon_buffer, lon_max + lon_buffer, n_absences),
            'month': np.random.randint(1, 13, n_absences),
            'year': np.random.choice(presences['year'].dropna().unique(), n_absences),
            'presence': 0
        })
        
        # Add simulated environmental features for absences
        absences['season'] = absences['month'].apply(self._get_season_southern_hemisphere)
        absences['day_of_year'] = (absences['month'] - 1) * 30 + 15
        
        # Simulate environmental conditions for absence points
        base_sst = 18 - (np.abs(absences['latitude']) - 30) * 0.5
        season_effect = absences['season'].map({
            'summer': 3, 'autumn': 0, 'winter': -3, 'spring': 0
        })
        absences['sst'] = base_sst + season_effect + np.random.normal(0, 1, len(absences))
        absences['sst'] = absences['sst'].clip(10, 30)
        
        coastal_distance = np.abs(absences['longitude'] - 25)
        absences['chlorophyll'] = 1.5 * np.exp(-coastal_distance / 5) + np.random.uniform(0.05, 0.2, len(absences))
        absences['chlorophyll'] = absences['chlorophyll'].clip(0.01, 10)
        
        absences['depth'] = 50 + coastal_distance * 20 + np.random.uniform(0, 100, len(absences))
        absences['depth'] = absences['depth'].clip(0, 5000)
        
        absences['distance_to_coast'] = coastal_distance * 10
        absences['sst_gradient'] = np.abs(np.random.normal(0, 0.5, len(absences)))
        
        # Add species column (NA for absences)
        absences['species'] = 'absence'
        
        print(f"✓ Generated {len(absences)} pseudo-absence points")
        
        return presences, absences
    
    def prepare_feature_matrix(self, target_species=None):
        """Create final feature matrix for modeling"""
        print("\n" + "=" * 80)
        print("PREPARING FEATURE MATRIX")
        print("=" * 80)
        
        presences, absences = self.generate_pseudo_absences(ratio=1.0)
        
        if target_species:
            print(f"Filtering for species: {target_species}")
            presences = presences[presences['species'] == target_species].copy()
            print(f"  Presence points: {len(presences)}")
        
        # Combine presences and absences
        combined = pd.concat([presences, absences], ignore_index=True)
        
        # Select features for modeling
        feature_cols = [
            'latitude', 'longitude', 
            'sst', 'chlorophyll', 'depth', 'distance_to_coast', 'sst_gradient',
            'month', 'day_of_year'
        ]
        
        # Create feature matrix
        X = combined[feature_cols].copy()
        y = combined['presence'].values
        
        # Store metadata
        metadata = combined[['latitude', 'longitude', 'species', 'presence']].copy()
        
        print(f"\n✓ Feature matrix prepared")
        print(f"  Total samples: {len(X)}")
        print(f"  Presence: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"  Absence: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
        print(f"  Features: {list(X.columns)}")
        print(f"\nFeature statistics:")
        print(X.describe())
        
        self.feature_matrix = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'feature_names': feature_cols
        }
        
        return X, y, metadata

if __name__ == "__main__":
    # Test the data preparation
    prep = SharkDataPreparation()
    prep.load_shark_data()
    prep.filter_shark_species(min_samples=50)
    prep.extract_basic_features()
    prep.add_environmental_features_simulated()
    
    # Prepare for main species
    if len(prep.species_list) > 0:
        target_species = prep.species_list[0]  # Use most common species
        print(f"\n{'='*80}")
        print(f"PREPARING DATA FOR: {target_species}")
        print('='*80)
        X, y, metadata = prep.prepare_feature_matrix(target_species=target_species)
        
        print("\n✓ Data preparation complete!")
        print(f"  Ready for Random Forest training")
