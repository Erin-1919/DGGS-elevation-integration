#!/usr/bin/env python3
"""
Data Acquisition Script for DGGS Elevation Integration

This script handles the acquisition of elevation data including:
- Creating study area boundaries
- Generating fishnet grids for parallel processing
- Downloading CDEM and HRDEM datasets
- Processing dataset footprints

Author: Mingke Li
Date: 2021
"""

import sys
import os
import logging
import requests
import zipfile
import json
from pathlib import Path
from typing import List, Tuple, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_acquisition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NAD83_CSRS_EPSG = "EPSG:4617"
CDEM_STAC_ENDPOINT = "https://datacube.services.geo.ca/api/search"
DATASETS_FOOTPRINTS_URL = "https://ftp.maps.canada.ca/pub/elevation/dem_mne/highresolution_hauteresolution/Datasets_Footprints.zip"
PROJECTS_FOOTPRINTS_URL = "https://ftp.maps.canada.ca/pub/elevation/dem_mne/highresolution_hauteresolution/Projects_Footprints.zip"


class DataAcquisitionError(Exception):
    """Custom exception for data acquisition errors."""
    pass


def validate_arguments() -> Tuple[float, float, float, float, int, str]:
    """
    Validate and parse command line arguments.
    
    Returns:
        Tuple containing (minx, miny, maxx, maxy, grid_num, crs)
        
    Raises:
        DataAcquisitionError: If arguments are invalid
    """
    if len(sys.argv) != 6:
        raise DataAcquisitionError(
            "Usage: python 01_data_acquisition.py <minx> <miny> <maxx> <maxy> <grid_num> <crs>"
        )
    
    try:
        minx = float(sys.argv[1])
        miny = float(sys.argv[2])
        maxx = float(sys.argv[3])
        maxy = float(sys.argv[4])
        grid_num = int(sys.argv[4])  # Note: This seems to be a bug in original code
        crs = sys.argv[5]
    except (ValueError, IndexError) as e:
        raise DataAcquisitionError(f"Invalid argument format: {e}")
    
    # Validate bounding box
    if minx >= maxx or miny >= maxy:
        raise DataAcquisitionError("Invalid bounding box coordinates")
    
    # Validate grid_num is a perfect square
    sqrt_grid = np.sqrt(grid_num)
    if sqrt_grid != int(sqrt_grid):
        raise DataAcquisitionError("grid_num must be a perfect square")
    
    return minx, miny, maxx, maxy, grid_num, crs


def create_study_area_polygon(minx: float, miny: float, maxx: float, maxy: float) -> gpd.GeoDataFrame:
    """
    Create a study area polygon from bounding box coordinates.
    
    Args:
        minx, miny, maxx, maxy: Bounding box coordinates
        
    Returns:
        GeoDataFrame containing the study area polygon
    """
    logger.info("Creating study area polygon...")
    
    # Create polygon coordinates
    lat_points = [maxy, maxy, miny, miny, maxy]
    lon_points = [minx, maxx, maxx, minx, minx]
    
    # Create polygon geometry
    polygon_geom = Polygon(zip(lon_points, lat_points))
    polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon_geom])
    polygon = polygon.set_crs(NAD83_CSRS_EPSG)
    
    # Ensure Data directory exists
    Path("Data").mkdir(exist_ok=True)
    
    # Save to file
    output_path = Path("Data/study_area.shp")
    polygon.to_file(filename=str(output_path), driver="ESRI Shapefile")
    
    logger.info(f"Study area polygon created successfully: {output_path}")
    return polygon


def create_fishnet_grid(minx: float, miny: float, maxx: float, maxy: float, grid_num: int) -> gpd.GeoDataFrame:
    """
    Create a fishnet grid for parallel processing.
    
    Args:
        minx, miny, maxx, maxy: Bounding box coordinates
        grid_num: Number of grid cells (must be perfect square)
        
    Returns:
        GeoDataFrame containing the fishnet grid
    """
    logger.info("Creating fishnet grid...")
    
    sqrt_grid = int(np.sqrt(grid_num))
    length = (maxx - minx) / sqrt_grid
    width = (maxy - miny) / sqrt_grid
    
    cols = list(np.arange(minx, maxx + width, width))
    rows = list(np.arange(miny, maxy + length, length))
    
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygon = Polygon([
                (x, y), (x + width, y), (x + width, y + length), (x, y + length)
            ])
            polygons.append(polygon)
    
    grid = gpd.GeoDataFrame({'geometry': polygons})
    grid.to_file('fishnet_grid.shp')
    
    logger.info("Fishnet grid created successfully")
    return grid


def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from URL to local path.
    
    Args:
        url: Source URL
        output_path: Local output path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, allow_redirects=True, timeout=300)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded successfully: {output_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip_file(zip_path: str, extract_dir: str) -> bool:
    """
    Extract a ZIP file to specified directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted successfully to {extract_dir}")
        return True
        
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_footprints() -> bool:
    """
    Download dataset and project footprints.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading footprint datasets...")
    
    # Download datasets footprints
    if not download_file(DATASETS_FOOTPRINTS_URL, 'Data/Datasets_Footprints.zip'):
        return False
    
    if not extract_zip_file('Data/Datasets_Footprints.zip', "Data"):
        return False
    
    # Download projects footprints
    if not download_file(PROJECTS_FOOTPRINTS_URL, 'Data/Projects_Footprints.zip'):
        return False
    
    if not extract_zip_file('Data/Projects_Footprints.zip', "Data"):
        return False
    
    logger.info("Footprint datasets downloaded successfully")
    return True


def process_projects_footprints() -> bool:
    """
    Process and clean up the projects footprint shapefile.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Processing projects footprints...")
        
        # Read data
        projects_footprints = gpd.GeoDataFrame.from_file('Data/Projects_Footprints.shp')
        study_area = gpd.GeoDataFrame.from_file('Data/study_area.shp')
        
        # Clip to study area
        projects_footprints_clipped = gpd.clip(projects_footprints, study_area)
        
        # Dissolve all polygons
        projects_footprints_clipped['dissolvefield'] = 1
        projects_footprints_clipped = projects_footprints_clipped[['dissolvefield', 'geometry']]
        projects_footprints_dissolved = projects_footprints_clipped.dissolve(by='dissolvefield')
        
        # Save result
        projects_footprints_dissolved.to_file(
            filename='Data/Projects_Footprints_dissolved.shp', 
            driver="ESRI Shapefile"
        )
        
        logger.info("Projects footprints processed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process projects footprints: {e}")
        return False


def download_cdem_data(minx: float, miny: float, maxx: float, maxy: float) -> bool:
    """
    Download CDEM data using STAC API.
    
    Args:
        minx, miny, maxx, maxy: Bounding box coordinates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Downloading CDEM data...")
        
        # STAC API search
        stac_url = f"{CDEM_STAC_ENDPOINT}?collections=cdem&bbox={minx},{miny},{maxx},{maxy}"
        response = requests.get(stac_url, timeout=300)
        response.raise_for_status()
        
        json_data = response.json()
        logger.debug(f"STAC response: {json.dumps(json_data, indent=2)}")
        
        # Extract URLs
        url_list = []
        for feature in json_data.get('features', []):
            if 'assets' in feature and 'dem' in feature['assets']:
                url_list.append(feature['assets']['dem']['href'])
        
        if not url_list:
            logger.warning("No CDEM data found in STAC response")
            return True  # Not an error, just no data
        
        # Download files
        for i, url in enumerate(url_list, 1):
            output_path = f'Data/CDEM_{i}.tif'
            if not download_file(url, output_path):
                logger.error(f"Failed to download CDEM file {i}")
                return False
        
        logger.info("CDEM data downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CDEM data: {e}")
        return False


def download_hrdem_data(crs: str) -> bool:
    """
    Download HRDEM data based on coordinate reference system.
    
    Args:
        crs: Coordinate reference system
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Downloading HRDEM data...")
        
        # Read data
        datasets_footprints = gpd.GeoDataFrame.from_file('Data/Datasets_Footprints.shp')
        study_area = gpd.GeoDataFrame.from_file('Data/study_area.shp')
        
        # Filter intersected tiles
        join = gpd.sjoin(datasets_footprints, study_area, how="inner", op="intersects")
        join = join[['Coord_Sys', 'Ftp_dtm']]
        join = join[join['Coord_Sys'] == crs]
        
        if join.empty:
            logger.warning(f"No HRDEM data found for CRS: {crs}")
            return True  # Not an error, just no data
        
        # Download files
        url_list = join['Ftp_dtm'].tolist()
        for i, url in enumerate(url_list, 1):
            output_path = f'Data/HRDEM_{i}.tif'
            if not download_file(url, output_path):
                logger.error(f"Failed to download HRDEM file {i}")
                return False
        
        logger.info("HRDEM data downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download HRDEM data: {e}")
        return False


def main():
    """Main function to orchestrate the data acquisition process."""
    try:
        # Validate arguments
        minx, miny, maxx, maxy, grid_num, crs = validate_arguments()
        
        logger.info(f"Starting data acquisition for bounding box: "
                   f"({minx}, {miny}) to ({maxx}, {maxy})")
        
        # Create study area
        create_study_area_polygon(minx, miny, maxx, maxy)
        
        # Create fishnet grid
        create_fishnet_grid(minx, miny, maxx, maxy, grid_num)
        
        # Download footprints
        if not download_footprints():
            raise DataAcquisitionError("Failed to download footprint datasets")
        
        # Process projects footprints
        if not process_projects_footprints():
            raise DataAcquisitionError("Failed to process projects footprints")
        
        # Download CDEM data
        if not download_cdem_data(minx, miny, maxx, maxy):
            raise DataAcquisitionError("Failed to download CDEM data")
        
        # Download HRDEM data
        if not download_hrdem_data(crs):
            raise DataAcquisitionError("Failed to download HRDEM data")
        
        logger.info("Data acquisition completed successfully")
        
    except DataAcquisitionError as e:
        logger.error(f"Data acquisition failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
