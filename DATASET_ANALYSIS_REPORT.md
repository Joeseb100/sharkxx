# Oceanographic Dataset Analysis Report
**Workspace:** d:\sharkxx  
**Analysis Date:** October 4, 2025

---

## Executive Summary

This workspace contains four distinct types of oceanographic and marine biology datasets:

1. **Chlorophyll Concentration Data** (11 files) - MODIS satellite imagery
2. **Sea Surface Temperature Data** (10 files) - GOES-16 satellite thermal imagery  
3. **Sea Surface Height/Wave Data** (6 files) - SWOT satellite altimetry
4. **Marine Species Occurrence Data** (3 datasets) - Shark tracking observations

---

## 1. Chlorophyll (CHL) Dataset Analysis

### Dataset Details
- **Source:** AQUA MODIS Level-3 Standard Mapped Images
- **Parameter:** Chlorophyll-a concentration (mg m⁻³)
- **Algorithm:** OCI (Ocean Color Index)

### Temporal Coverage
- **Date Range:** August 27-31, 2025 (5 days)
- **Temporal Resolution:** Daily composites

### Spatial Coverage
- **Global Coverage:** Latitude: -90° to +90°, Longitude: -180° to +180°
- **Spatial Resolutions:** 
  - 4 km (4.638 km): 4320 × 8640 pixels
  - 9 km: Lower resolution available

### Data Quality & Statistics
**Sample Analysis (Aug 27, 2025 - 4km resolution):**
- **Total Pixels:** 37,324,800
- **Valid Data Points:** 2,662,011 (7.1% coverage)
- **Missing/Land/Cloud:** 34,662,789 (92.9%)
- **Chlorophyll-a Range:** 0.001 to 86.374 mg m⁻³
- **Mean:** 0.613 mg m⁻³
- **Median:** 0.089 mg m⁻³

### Key Observations
- High data gaps typical of optical satellite sensors (clouds, land, ice)
- Wide range of chlorophyll values indicates diverse marine environments
- Median much lower than mean suggests right-skewed distribution (typical for ocean chlorophyll)
- Values >1 mg m⁻³ indicate productive waters (coastal/upwelling zones)

### Files
```
AQUA_MODIS.20250827.L3m.DAY.CHL.chlor_a.4km.nc (Aug 27)
AQUA_MODIS.20250827.L3m.DAY.CHL.chlor_a.9km.nc (Aug 27)
AQUA_MODIS.20250828.L3m.DAY.CHL.chlor_a.4km.nc (Aug 28)
AQUA_MODIS.20250828.L3m.DAY.CHL.chlor_a.9km.nc (Aug 28)
AQUA_MODIS.20250829.L3m.DAY.CHL.chlor_a.4km.nc (Aug 29)
AQUA_MODIS.20250829.L3m.DAY.CHL.chlor_a.9km.nc (Aug 29)
AQUA_MODIS.20250830.L3m.DAY.CHL.chlor_a.4km.nc (Aug 30)
AQUA_MODIS.20250830.L3m.DAY.CHL.chlor_a.9km.nc (Aug 30) [+ duplicate]
AQUA_MODIS.20250831.L3m.DAY.CHL.chlor_a.4km.nc (Aug 31)
AQUA_MODIS.20250831.L3m.DAY.CHL.chlor_a.9km.nc (Aug 31)
```

---

## 2. Sea Surface Temperature (SST) Dataset Analysis

### Dataset Details
- **Source:** GOES-16 ABI (Advanced Baseline Imager)
- **Product:** GHRSST L2P SST (Group for High Resolution SST, Level 2 Preprocessed)
- **Parameter:** Sea surface sub-skin temperature
- **Platform:** GOES-16 (Geostationary satellite)
- **Algorithm:** ACSPO V2.70

### Temporal Coverage
- **Date:** April 3, 2025
- **Time Range:** 08:00:00 to 11:00:00 UTC (4 hours)
- **Temporal Resolution:** Hourly observations

### Spatial Coverage
- **Latitude:** -81.15° to +81.15°
- **Longitude:** -156.20° to +6.20° (Western Hemisphere - Atlantic & Pacific)
- **Spatial Resolution:** 2 km at nadir
- **Grid Size:** 5424 × 5424 pixels

### Data Quality & Statistics
**Sample Analysis (April 3, 2025 - 08:00 UTC):**
- **Total Pixels:** 29,419,776
- **Valid SST Points:** 3,654,750 (12.4% coverage)
- **Temperature Range:** 271.05 K to 303.89 K (-2.1°C to 30.7°C)
- **Mean Temperature:** 298.42 K (25.3°C)

### Additional Variables
The dataset includes rich auxiliary data:
- Multiple brightness temperature channels (8.6, 10.4, 11.2, 12.3 μm)
- Satellite zenith angle
- Wind speed
- Sea ice fraction
- Quality flags and uncertainty estimates
- SSES (Sensor-Specific Error Statistics) bias and standard deviation

### Key Observations
- Geostationary coverage provides frequent temporal sampling
- Temperature range spans from polar to tropical waters
- Mean temperature of 25.3°C suggests tropical/subtropical focus
- Low data coverage (12.4%) typical for cloudy conditions

### Files
```
20250403080000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70 (08:00 UTC)
20250403090000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70 (09:00 UTC)
20250403100000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70 (10:00 UTC)
20250403110000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70 (11:00 UTC)
+ 2 misplaced chlorophyll files
```

**Note:** Two AQUA_MODIS chlorophyll files are incorrectly placed in the SST directory.

---

## 3. Wave/SSH Dataset Analysis (SWOT)

### Dataset Details
- **Mission:** SWOT (Surface Water and Ocean Topography)
- **Product Types:**
  - Basic SSH: Standard sea surface height
  - Expert SSH: Additional processing/corrections
  - Unsmoothed: Raw measurements
  - WindWave: Wave-specific parameters
- **Level:** L2 LR (Level 2 Low Rate)

### Temporal Coverage
- **Date:** October 1, 2025
- **Time Range:** 21:55:14 to 23:37:27 UTC (~1.7 hours)
- **Two orbital passes:** Pass 304 and 305

### Spatial Coverage
- **Latitude:** -78.27° to +78.27° (near-polar coverage)
- **Longitude:** 134.07° to 301.51° (Pacific Ocean crossing)
- **Swath Dimensions:** 9,866 lines × 69 pixels × 2 sides

### Key Variables
**SSH (Sea Surface Height) Products:**
- `ssh_karin`: Primary sea surface height measurement
- `ssha_karin`: Sea surface height anomaly (relative to mean)
- Quality flags and uncertainty estimates
- Multiple SSH products with different processing

**Auxiliary Data:**
- Distance and heading to coast
- Surface classification (water, ice, land)
- Dynamic ice flag
- Rain flag (affects radar measurements)
- Radiometer surface type

### Key Observations
- High-resolution altimetry with ~2 km across-track resolution
- Swath coverage provides detailed 2D SSH mapping (unlike traditional nadir altimeters)
- Multiple processing levels allow quality assessment
- Includes environmental flags critical for data quality control

### Files
```
SWOT_L2_LR_SSH_Basic_039_304_... (Pass 304, 21:55-22:46 UTC)
SWOT_L2_LR_SSH_Basic_039_305_... (Pass 305, 22:46-23:37 UTC)
SWOT_L2_LR_SSH_Expert_039_305_... (Pass 305, Expert product)
SWOT_L2_LR_SSH_Unsmoothed_039_305_... (Pass 305, Unsmoothed)
SWOT_L2_LR_SSH_WindWave_039_304_... (Pass 304, Wave parameters)
```

---

## 4. Marine Species Occurrence Data Analysis

### Dataset Details
- **Source:** GBIF (Global Biodiversity Information Facility)
- **Data Type:** Marine species occurrence records
- **Primary Focus:** Shark tracking data

### Temporal Coverage
- **Date Range:** March 6, 2013 to November 15, 2021 (~8.7 years)

### Spatial Coverage
**Sample Dataset (0046525-250920141307145):**
- **Latitude:** -34.73° to -26.85° (Southern Africa)
- **Longitude:** 18.29° to 32.90° (Indian Ocean to Atlantic)
- **Region:** South African coastal waters

### Species Information
**Dataset 0046525-250920141307145:**
- **Total Records:** 2,864 observations
- **Unique Species:** 2
  1. **Notorynchus cepedianus** (Broadnose sevengill shark): 2,789 records (97.4%)
  2. **Carcharodon carcharias** (Great white shark): 75 records (2.6%)

**Conservation Status:**
- IUCN Red List Category: **VU (Vulnerable)** for both species

### Data Structure
- **Total Columns:** 226 attributes per record
- **Key Fields:**
  - Geographic coordinates (decimalLatitude, decimalLongitude)
  - Temporal information (eventDate)
  - Species identification (species, scientificName)
  - GBIF identifiers (gbifID)
  - Access rights and licenses
  - Conservation status
  - Data source: Ocean Tracking Network

### Key Observations
- Long-term tracking data ideal for migration and behavior studies
- Focused on South African waters (important shark habitat)
- Heavy bias toward one species (Broadnose sevengill shark)
- Data from acoustic tracking network (OTN - Ocean Tracking Network)
- Both species are vulnerable, highlighting conservation importance

### Datasets
```
0046525-250920141307145/ (2,864 records - Sevengill & Great White sharks)
0046528-250920141307145/ (Not yet analyzed)
0046539-250920141307145/ (Not yet analyzed)
```

---

## Cross-Dataset Integration Opportunities

### 1. **Shark Habitat Characterization**
Combine species occurrence data with environmental parameters:
- **SST**: Analyze temperature preferences and thermal boundaries
- **CHL**: Investigate relationship with primary productivity and food web
- **SSH**: Examine associations with oceanographic features (eddies, fronts)

### 2. **Temporal Analysis Challenges**
- **Major gap:** Occurrence data (2013-2021) vs. satellite data (2025)
- **Recommendation:** Acquire historical satellite data or recent shark tracking data

### 3. **Spatial Overlap Analysis**
- **Limited overlap:** South African waters (occurrence) vs. broader satellite coverage
- **Opportunity:** Extract satellite data for specific shark locations
- **Regional focus:** South African coast (18-33°E, -35 to -27°N)

### 4. **Multi-Sensor Analysis**
- **High-frequency SST** (hourly) for thermal habitat tracking
- **Daily chlorophyll** for productivity monitoring
- **SWOT SSH** for mesoscale oceanographic features

---

## Data Quality Issues

### 1. **File Organization**
- ⚠️ Two chlorophyll files misplaced in `sst/` directory
- ⚠️ Duplicate files with "(1)" suffix indicate potential download issues

### 2. **Temporal Gaps**
- 5-month gap between SST (April) and CHL (August)
- SWOT data from October (different month than both)
- 4+ year gap between occurrence data and satellite observations

### 3. **Data Coverage**
- Low valid pixel rates in satellite data (7-12%) due to clouds
- Species occurrence limited to 2 species in examined dataset

---

## Recommendations

### For Data Management
1. **Reorganize files:** Move misplaced chlorophyll files to correct directory
2. **Remove duplicates:** Delete files with "(1)" suffix after verification
3. **Create metadata:** Document acquisition dates, sources, and purposes

### For Analysis
1. **Extract regional data:** Subset satellite data to South African shark habitat region
2. **Time series analysis:** Analyze environmental conditions at shark detection locations
3. **Acquire continuous data:** Fill temporal gaps with additional satellite acquisitions
4. **Explore other datasets:** Analyze the two unexamined occurrence datasets

### For Shark-Environment Studies
1. **Match coordinates:** Extract SST and chlorophyll at shark occurrence locations
2. **Habitat modeling:** Use environmental variables to predict shark presence
3. **Movement corridors:** Analyze SSH features along migration routes
4. **Seasonal patterns:** Compare environmental preferences across seasons

---

## Technical Specifications Summary

| Dataset | Format | Resolution | Coverage | Variables |
|---------|--------|------------|----------|-----------|
| Chlorophyll | NetCDF4 | 4 km & 9 km | Global | chlor_a + palette |
| SST | NetCDF4 | 2 km (nadir) | W. Hemisphere | SST + 15 aux vars |
| SWOT SSH | NetCDF4 | ~2 km swath | Near-polar | SSH + quality flags |
| Occurrences | Tab-delimited | Point data | S. Africa | 226 attributes |

---

## Conclusion

This workspace contains a rich but temporally fragmented collection of oceanographic data with strong potential for integrated marine biology research. The primary scientific opportunity lies in characterizing shark habitat preferences using multi-sensor satellite observations. However, the 4+ year temporal gap between biological and physical data represents a significant limitation that should be addressed through data acquisition before comprehensive analysis.

**Primary Use Case Identified:** Shark habitat characterization and environmental preference analysis for vulnerable species in South African waters.

**Critical Next Step:** Acquire historical satellite data (2013-2021) matching the shark observation period, or obtain recent shark tracking data (2025) to match satellite observations.
