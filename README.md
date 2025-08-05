# Integration of Heterogeneous Terrain Data into Discrete Global Grid Systems -- Source Code

This work aimed to integrate multi-source terrain data in Canada by adopting the ISEA3H DGGS. The modeling process of terrain data in the ISEA3H DGGS had the following main phases: data acquisition, pre-processing, quantization, aggregation, and quality control. The open-sourced library [*dggridR*](https://github.com/r-barnes/dggridR) was used to complete conversion between geographic locations and ISEA3H DGGS cell indices. The modeling process was developed using a hybrid of Python 3.7.7 and R 3.6.2 environments. The code used to conduct the experiment are available in the folder [*Script*](https://github.com/Erin-1919/DGGS-Elevation-Integration/tree/main/Script).

## Recent Updates (2024)
The codebase has been professionally refactored with the following improvements:
- **Enhanced Error Handling**: Comprehensive exception handling and logging throughout all scripts
- **Professional Code Structure**: Proper imports, type hints, docstrings, and modular design
- **Improved Documentation**: Detailed function documentation and inline comments
- **Better Configuration Management**: Constants, lookup tables, and environment variable handling
- **Memory Management**: Proper resource cleanup and garbage collection
- **HPC Compatibility**: Maintained SLURM integration and parallel processing capabilities
- **Logging System**: Comprehensive logging with both file and console output

## Manuscript Information
### Title of Manuscript
Integration of Heterogeneous Terrain Data into Discrete Global Grid Systems

### Keywords
Discrete Global Grid Systems, terrain data, data integration, national elevation service, multi-resolution elevation, parallel computing

### DOI
10.1080/15230406.2021.1966648

### Authors
Mingke Li, Heather McGrath, and Emmanuel Stefanakis

### Corresponding Author
[Mingke Li](https://erin-1919.github.io/) (mingke.li@ucalgary.ca)

[ORCID](https://orcid.org/0000-0001-6310-4964)

### Abstract
The Canadian Digital Elevation Model (CDEM) and the High Resolution Digital Elevation Model (HRDEM) released by Natural Resources Canada are primary terrain data sources in Canada. Due to their different coverage, datums, resolutions, and accuracies, a standardized framework for national elevation data across various scales is required. This study provides new insights into the adoption of Discrete Global Grid Systems (DGGS) to facilitate the integration of multi-source terrain data at various granularities. In particular, the Icosahedral Snyder Equal Area Aperture 3 Hexagonal Grid (ISEA3H) was employed, and quantization, integration, and aggregation were conducted on this framework. To demonstrate the modeling process, an experiment was undertaken for two areas in Ontario, taking advantage of parallel computing which was beneficial from the discreteness of DGGS cells. The accuracy of the modeled elevations was estimated by referring to the ground-surveyed values and was included in the spatially referenced metadata as an indicator of data quality. This research can serve as a guide for future development of a national elevation service, providing consistent, multi-resolution elevations and avoiding complex, duplicated pre-processing at the user's end. Future investigation into an operational integration platform to support real-world decision-making, as well as the DGGS-powered geospatial datacube, is recommended.

### Code Repository
https://github.com/Erin-1919/DGGS-Elevation-Integration

## Script Overview

The processing pipeline consists of 10 scripts that handle different stages of the DGGS elevation integration:

### Data Preparation Scripts
1. **01_data_acquisition.py** - Downloads CDEM and HRDEM data, creates study area boundaries and fishnet grids
2. **02_cdem_preprocess.py** - Mosaics CDEM tiles, converts to NAD83 CSRS, and transforms vertical datum to CGVD2013
3. **03_hrdem_preprocess.py** - Reprojects HRDEM to geographic coordinates, mosaics tiles, and crops to study area

### DGGS Processing Scripts
4. **04_generate_centroids.R** - Generates DGGS cell centroids for parallel processing
5. **05_dggs_modeling.py** - Extracts elevation values from DEMs using DGGS interpolation
6. **06_dggs_navigation.R** - Creates parent-child relationships between DGGS resolution levels
7. **07_dggs_aggregation.py** - Calculates elevation statistics (mean, max, min) for coarse resolution cells

### Validation and Visualization Scripts
8. **08_control_points.py** - Validates DGGS modeling using ground control points
9. **09_visualization_quanti.py** - Creates visualizations of quantization results
10. **10_visualization_aggre.py** - Creates visualizations of aggregation statistics

## Usage Instructions

### Prerequisites
- Python 3.7+ with required libraries (see below)
- R 3.6+ with required packages (see below)
- Access to Canadian elevation data sources
- HPC environment with SLURM (optional, for parallel processing)

### Installation
```bash
# Clone the repository
git clone https://github.com/Erin-1919/DGGS-Elevation-Integration.git
cd DGGS-Elevation-Integration

# Install Python dependencies
pip install -r requirements.txt

# Install R packages
R -e "install.packages(c('dggridR', 'rgdal', 'rgeos', 'dplyr', 'doParallel', 'tictoc', 'logging'))"
```

### HPC Job Submission
Sample SLURM job scripts are provided in the `Sample_job_script/` folder for running the pipeline on high-performance computing clusters.

## Libraries Used

### Python Dependencies
- requests 2.24.0
- geopandas 0.9.0
- shapely 1.7.1
- pyRserve 0.9.2
- datashader 0.12.0
- matplotlib 3.3.2
- numpy 1.19.4
- scipy 1.5.3
- rasterio 1.2.1
- gdal 3.1.4
- pandas 1.1.4
- multiprocess 0.70.12.2

### R Dependencies
- dggridR 2.0.4
- rgdal 1.5.16
- rgeos 0.5.5
- dplyr 1.0.2
- doParallel 1.0.16
- tictoc (for timing)
- logging (for error handling)

## Data Availability
The original Canadian Digital Elevation Model (CDEM) data can be downloaded via the Geospatial-Data Extraction tool in [Canada's Open Government Portal](https://maps.canada.ca/czs/index-en.html), or they can be obtained through the STAC API as shown in the [sample code](https://github.com/Erin-1919/DGGS-Elevation-Integration/blob/main/Script/01_data_acquisition.py). The High Resolution Digital Elevation Model (HRDEM) data are available at the [data repository](https://ftp.maps.canada.ca/pub/elevation/dem_mne/highresolution_hauteresolution/dtm_mnt). Ground control points are accessible using [COSINE](https://www.lioapplications.lrc.gov.on.ca/COSINE/index.html?viewer=COSINE.OntarioViewer&locale=en-CA), Ontario's geodetic control database, on the website of [Ontario Ministry of Natural Resources and Forestry](https://www.ontario.ca/page/geodesy). The conversion grids between CGVD28 and CGVD2013 is stored in the folder [*Data*](https://github.com/Erin-1919/DGGS-Elevation-Integration/tree/main/Data), and it was originally downloaded from the website of [Natural Resources Canada (NRCan)](https://webapp.geod.nrcan.gc.ca/geod/process/download-helper.php?file_id=HT2_2010_CGG2013a), where login is needed. The experimental data, including the study areas (.shp), fishnet grids (.shp), and ground control points (.csv) are available in the folder [*Experiment_data*](https://github.com/Erin-1919/DGGS-Elevation-Integration/tree/main/Experiment_data).

## Experiment Note
The experiment was carried out on the Advanced Research Computing cluster at the University of Calgary. To improve the computational efficiency and take advantage of the discrete property of DGGS cells, the modeling process ran in a parallelism fashion, which was a hybrid of the shared-memory parallelism and the job-level parallelism. Sample job scripts sent to the high-performance cluster are provided in the folder [*Sample_job_script*](https://github.com/Erin-1919/DGGS-Elevation-Integration/tree/main/Sample_job_script).
