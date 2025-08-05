#!/usr/bin/env python3
"""
DGGS Modeling Script for Elevation Integration

This script performs DGGS modeling by extracting elevation values with interpolation:
- Reads centroid data from CSV files
- Extracts elevation values from CDEM and HRDEM rasters
- Performs interpolation for missing values
- Handles different data sources based on availability

Author: Mingke Li
Date: 2021
"""

import sys
import os
import time
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Union

import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import multiprocess as mp
from scipy import interpolate

# Enable shapely speedups
shapely.speedups.enable()
warnings.simplefilter('error', RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dggs_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NO_DATA_VALUE = -32767.0
NAD83_CSRS_EPSG = "EPSG:4617"

# Resolution lookup table
RESOLUTION_LOOKUP = {
    16: {'cell_size': 0.005, 'vertical_res': 0},
    17: {'cell_size': 0.003, 'vertical_res': 0},
    18: {'cell_size': 0.001, 'vertical_res': 0},
    19: {'cell_size': 0.0009, 'vertical_res': 1},
    20: {'cell_size': 0.0008, 'vertical_res': 1},
    21: {'cell_size': 0.0003, 'vertical_res': 2},
    22: {'cell_size': 0.0003, 'vertical_res': 2},
    23: {'cell_size': 0.0001, 'vertical_res': 3},
    24: {'cell_size': 0.0001, 'vertical_res': 3},
    25: {'cell_size': 0.00006, 'vertical_res': 4},
    26: {'cell_size': 0.00003, 'vertical_res': 4},
    27: {'cell_size': 0.00002, 'vertical_res': 5},
    28: {'cell_size': 0.00001, 'vertical_res': 5},
    29: {'cell_size': 0.000005, 'vertical_res': 6}
}


class DGGSModelingError(Exception):
    """Custom exception for DGGS modeling errors."""
    pass


def validate_arguments() -> Tuple[int, int]:
    """
    Validate and parse command line arguments.
    
    Returns:
        Tuple containing (dggs_res, fid)
        
    Raises:
        DGGSModelingError: If arguments are invalid
    """
    if len(sys.argv) < 2:
        raise DGGSModelingError("Usage: python 05_dggs_modeling.py <dggs_resolution>")
    
    try:
        dggs_res = int(sys.argv[1])
        fid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    except (ValueError, TypeError) as e:
        raise DGGSModelingError(f"Invalid argument format: {e}")
    
    if dggs_res not in RESOLUTION_LOOKUP:
        raise DGGSModelingError(f"Invalid DGGS resolution: {dggs_res}")
    
    return dggs_res, fid


def get_resolution_params(dggs_res: int) -> Tuple[float, int]:
    """
    Get cell size and vertical resolution for given DGGS resolution.
    
    Args:
        dggs_res: DGGS resolution level
        
    Returns:
        Tuple containing (cell_size, vertical_res)
    """
    params = RESOLUTION_LOOKUP.get(dggs_res)
    if not params:
        raise DGGSModelingError(f"Invalid DGGS resolution: {dggs_res}")
    
    return params['cell_size'], params['vertical_res']


def load_input_data(dggs_res: int, fid: int) -> pd.DataFrame:
    """
    Load input CSV data containing centroid information.
    
    Args:
        dggs_res: DGGS resolution level
        fid: Grid ID
        
    Returns:
        DataFrame containing centroid data
    """
    input_path = f'Result/Level{dggs_res}/Centroid/vege_pre_{fid}.csv'
    
    if not os.path.exists(input_path):
        raise DGGSModelingError(f"Input file not found: {input_path}")
    
    try:
        df = pd.read_csv(input_path, usecols=['Cell_address', 'lon_c', 'lat_c'])
        logger.info(f"Loaded {len(df)} centroids from {input_path}")
        return df
    except Exception as e:
        raise DGGSModelingError(f"Failed to load input data: {e}")


def load_dem_data() -> Tuple[rasterio.DatasetReader, rasterio.DatasetReader, gpd.GeoDataFrame]:
    """
    Load DEM data and HRDEM extent.
    
    Returns:
        Tuple containing (CDEM_TIF, HRDEM_TIF, HRDEM_extent)
    """
    try:
        # Load DEMs
        cdem_tif = rasterio.open('Data/CDEM_cgvd2013.tif')
        hrdem_tif = rasterio.open('Data/HRDEM_cgvd2013.tif')
        
        # Load HRDEM extent
        hrdem_extent = gpd.GeoDataFrame.from_file('Data/Projects_Footprints_dissolved.shp')
        hrdem_extent = hrdem_extent.to_crs(NAD83_CSRS_EPSG)
        
        logger.info("DEM data loaded successfully")
        return cdem_tif, hrdem_tif, hrdem_extent
        
    except Exception as e:
        raise DGGSModelingError(f"Failed to load DEM data: {e}")


def find_neighbors(x: float, y: float, dem_tif: rasterio.DatasetReader) -> Tuple[List[float], List[float], List[float]]:
    """
    Find neighbors for interpolation.
    
    Args:
        x, y: Coordinates
        dem_tif: DEM raster dataset
        
    Returns:
        Tuple containing (x_array, y_array, z_array)
    """
    try:
        # Get pixel coordinates
        x_index, y_index = rasterio.transform.rowcol(dem_tif.transform, x, y)
        xc, yc = rasterio.transform.xy(dem_tif.transform, x_index, y_index)
        
        # Determine which quadrant the point falls in
        if x > xc and y > yc:
            x_index_array = [x_index-1, x_index-1, x_index, x_index]
            y_index_array = [y_index, y_index+1, y_index, y_index+1]
        elif x > xc and y < yc:
            x_index_array = [x_index, x_index, x_index+1, x_index+1]
            y_index_array = [y_index, y_index+1, y_index, y_index+1]
        elif x < xc and y > yc:
            x_index_array = [x_index-1, x_index-1, x_index, x_index]
            y_index_array = [y_index-1, y_index, y_index-1, y_index]
        elif x < xc and y < yc:
            x_index_array = [x_index, x_index, x_index+1, x_index+1]
            y_index_array = [y_index-1, y_index, y_index-1, y_index]
        else:
            # Point is exactly on a grid point
            x_index_array = [x_index, x_index, x_index, x_index]
            y_index_array = [y_index, y_index, y_index, y_index]
        
        # Convert to geographic coordinates
        x_array, y_array = rasterio.transform.xy(dem_tif.transform, x_index_array, y_index_array)
        
        # Extract elevation values
        coords = [(lon, lat) for lon, lat in zip(x_array, y_array)]
        z_array = [elev[0] for elev in dem_tif.sample(coords)]
        
        return x_array, y_array, z_array
        
    except Exception as e:
        logger.warning(f"Failed to find neighbors for ({x}, {y}): {e}")
        return [], [], []


def extract_cdem_elevation(x: float, y: float, cdem_tif: rasterio.DatasetReader, 
                          vertical_res: int, interp: str = 'linear') -> float:
    """
    Extract elevation from CDEM with interpolation.
    
    Args:
        x, y: Coordinates
        cdem_tif: CDEM raster dataset
        vertical_res: Vertical resolution for rounding
        interp: Interpolation method
        
    Returns:
        Interpolated elevation value
    """
    try:
        x_array, y_array, z_array = find_neighbors(x, y, cdem_tif)
        
        if not x_array or -32767 in z_array:
            return NO_DATA_VALUE
        
        # Perform interpolation
        cdem_interp = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
        elevation = cdem_interp(x, y)[0]
        
        # Round to specified precision
        elevation = round(elevation, vertical_res)
        
        return elevation
        
    except Exception as e:
        logger.debug(f"CDEM extraction failed for ({x}, {y}): {e}")
        return NO_DATA_VALUE


def extract_hrdem_elevation(x: float, y: float, hrdem_tif: rasterio.DatasetReader,
                           hrdem_extent: gpd.GeoDataFrame, vertical_res: int,
                           interp: str = 'linear') -> Optional[float]:
    """
    Extract elevation from HRDEM with interpolation.
    
    Args:
        x, y: Coordinates
        hrdem_tif: HRDEM raster dataset
        hrdem_extent: HRDEM extent polygon
        vertical_res: Vertical resolution for rounding
        interp: Interpolation method
        
    Returns:
        Interpolated elevation value or None if not available
    """
    try:
        # Check if point falls within HRDEM extent
        point = shapely.geometry.Point(x, y)
        if not point.within(hrdem_extent.geometry.iloc[0]):
            return None
        
        x_array, y_array, z_array = find_neighbors(x, y, hrdem_tif)
        
        if not x_array or -32767 in z_array:
            return None
        
        # Perform interpolation
        hrdem_interp = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
        elevation = hrdem_interp(x, y)[0]
        
        # Round to specified precision
        elevation = round(elevation, vertical_res)
        
        return elevation
        
    except Exception as e:
        logger.debug(f"HRDEM extraction failed for ({x}, {y}): {e}")
        return None


def process_dataframe_chunk(df_chunk: pd.DataFrame, cdem_tif: rasterio.DatasetReader,
                           hrdem_tif: rasterio.DatasetReader, hrdem_extent: gpd.GeoDataFrame,
                           vertical_res: int) -> pd.DataFrame:
    """
    Process a chunk of the dataframe to extract elevations.
    
    Args:
        df_chunk: DataFrame chunk to process
        cdem_tif: CDEM raster dataset
        hrdem_tif: HRDEM raster dataset
        hrdem_extent: HRDEM extent polygon
        vertical_res: Vertical resolution for rounding
        
    Returns:
        DataFrame with elevation columns added
    """
    try:
        # Extract CDEM elevations
        df_chunk['model_cdem'] = [
            extract_cdem_elevation(lon, lat, cdem_tif, vertical_res)
            for lon, lat in zip(df_chunk.lon_c, df_chunk.lat_c)
        ]
        
        # Extract HRDEM elevations
        df_chunk['model_hrdem'] = [
            extract_hrdem_elevation(lon, lat, hrdem_tif, hrdem_extent, vertical_res)
            for lon, lat in zip(df_chunk.lon_c, df_chunk.lat_c)
        ]
        
        return df_chunk
        
    except Exception as e:
        logger.error(f"Failed to process dataframe chunk: {e}")
        raise


def save_results(df: pd.DataFrame, dggs_res: int, fid: int) -> None:
    """
    Save results to CSV file.
    
    Args:
        df: DataFrame to save
        dggs_res: DGGS resolution
        fid: Grid ID
    """
    try:
        # Create output directory
        output_dir = Path(f"Result/Level{dggs_res}/Elev")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove coordinate columns for output
        df_output = df.drop(columns=['lon_c', 'lat_c'])
        
        # Save to CSV
        output_path = output_dir / f'vege_elev_{fid}.csv'
        df_output.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        raise DGGSModelingError(f"Failed to save results: {e}")


def main():
    """Main function to orchestrate DGGS modeling."""
    start_time = time.time()
    
    try:
        # Validate arguments
        dggs_res, fid = validate_arguments()
        logger.info(f"Starting DGGS modeling for resolution {dggs_res}, grid {fid}")
        
        # Get resolution parameters
        cell_size, vertical_res = get_resolution_params(dggs_res)
        logger.info(f"Cell size: {cell_size}, Vertical resolution: {vertical_res}")
        
        # Load input data
        fishnet_df = load_input_data(dggs_res, fid)
        
        # Load DEM data
        cdem_tif, hrdem_tif, hrdem_extent = load_dem_data()
        
        try:
            # Setup parallel processing
            n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
            logger.info(f"Using {n_cores} cores for parallel processing")
            
            # Split dataframe for parallel processing
            df_split = np.array_split(fishnet_df, n_cores)
            
            # Process in parallel
            with mp.Pool(processes=n_cores) as pool:
                results = pool.map(
                    lambda chunk: process_dataframe_chunk(
                        chunk, cdem_tif, hrdem_tif, hrdem_extent, vertical_res
                    ),
                    df_split
                )
            
            # Combine results
            fishnet_df_output = pd.concat(results, ignore_index=True)
            
            # Save results
            save_results(fishnet_df_output, dggs_res, fid)
            
            processing_time = time.time() - start_time
            logger.info(f"DGGS modeling completed in {processing_time:.2f} seconds")
            
        finally:
            # Clean up raster datasets
            cdem_tif.close()
            hrdem_tif.close()
        
    except DGGSModelingError as e:
        logger.error(f"DGGS modeling failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
