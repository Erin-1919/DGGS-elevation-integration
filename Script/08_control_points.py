#!/usr/bin/env python3
"""
Control Points Validation Script for DGGS Elevation Integration

This script validates DGGS modeling results using ground control points:
- Loads ground control point data
- Extracts pre-DGGS elevation values from DEMs
- Calculates post-DGGS elevation values using DGGS modeling
- Compares results across different resolution levels

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
import pyRserve
from scipy import interpolate

# Enable shapely speedups
shapely.speedups.enable()
warnings.simplefilter('error', RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('control_points.log'),
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


class ControlPointsError(Exception):
    """Custom exception for control points processing errors."""
    pass


def setup_r_connection() -> pyRserve.RConnection:
    """
    Setup R connection for DGGS operations.
    
    Returns:
        R connection object
    """
    try:
        conn = pyRserve.connect()
        
        # Import dggridR library
        conn.eval('library(dggridR)')
        
        # Define function to convert geographic coordinates to cell centroids
        conn.voidEval('''
        geo_to_centroid <- function(resolution, lon, lat) {
          v_lat = 37.6895
          v_lon = -51.6218
          azimuth = 360-72.6482
          DGG = dgconstruct(projection = "ISEA", aperture = 3, topology = "HEXAGON", res = resolution, 
                           precision = 7, azimuth_deg = azimuth, pole_lat_deg = v_lat, pole_lon_deg = v_lon)
          Cell_address = dgGEO_to_SEQNUM(DGG, lon, lat)$seqnum
          lon_c = dgSEQNUM_to_GEO(DGG, Cell_address)$lon_deg
          lat_c = dgSEQNUM_to_GEO(DGG, Cell_address)$lat_deg
          lon_lat = c(lon_c, lat_c)
          return(lon_lat)
        }
        ''')
        
        logger.info("R connection established successfully")
        return conn
        
    except Exception as e:
        raise ControlPointsError(f"Failed to setup R connection: {e}")


def load_dem_data() -> Tuple[rasterio.DatasetReader, rasterio.DatasetReader, gpd.GeoDataFrame]:
    """
    Load DEM data and HRDEM extent.
    
    Returns:
        Tuple containing (CDEM_TIF, HRDEM_TIF, HRDEM_extent)
    """
    try:
        # Load DEMs
        cdem_tif = rasterio.open('Data/CDEM_cgvd2013.tif')
        hrdem_tif = rasterio.open('Data/HRDEM_mosaic.tif')
        
        # Load HRDEM extent
        hrdem_extent = gpd.GeoDataFrame.from_file('Data/Projects_Footprints_dissolved.shp')
        hrdem_extent = hrdem_extent.to_crs(NAD83_CSRS_EPSG)
        
        logger.info("DEM data loaded successfully")
        return cdem_tif, hrdem_tif, hrdem_extent
        
    except Exception as e:
        raise ControlPointsError(f"Failed to load DEM data: {e}")


def check_point_in_hrdem_extent(x: float, y: float, hrdem_extent: gpd.GeoDataFrame) -> int:
    """
    Check if a point falls within HRDEM extent.
    
    Args:
        x, y: Point coordinates
        hrdem_extent: HRDEM extent polygon
        
    Returns:
        1 if point is within extent, 0 otherwise
    """
    try:
        point = shapely.geometry.Point(x, y)
        if point.within(hrdem_extent.geometry.iloc[0]):
            return 1
        else:
            return 0
    except Exception as e:
        logger.warning(f"Failed to check point ({x}, {y}) in HRDEM extent: {e}")
        return 0


def safe_extract_elevation(func, dem_type: str, *args, **kwargs) -> Union[float, None]:
    """
    Safely extract elevation values with error handling.
    
    Args:
        func: Function to call
        dem_type: Type of DEM ('CDEM' or 'HRDEM')
        *args, **kwargs: Arguments for the function
        
    Returns:
        Elevation value or None/NO_DATA_VALUE on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Elevation extraction failed for {dem_type}: {e}")
        if dem_type == 'CDEM':
            return NO_DATA_VALUE
        elif dem_type == 'HRDEM':
            return None
        else:
            return None


def extract_pre_dggs_elevation(dataframe: pd.DataFrame, cdem_tif: rasterio.DatasetReader,
                               hrdem_tif: rasterio.DatasetReader) -> pd.DataFrame:
    """
    Extract pre-DGGS elevation values from DEMs.
    
    Args:
        dataframe: DataFrame with control points
        cdem_tif: CDEM raster dataset
        hrdem_tif: HRDEM raster dataset
        
    Returns:
        DataFrame with elevation columns added
    """
    try:
        # Extract coordinates
        coords = [(lon, lat) for lon, lat in zip(dataframe.lon, dataframe.lat)]
        
        # Extract CDEM values
        dataframe['CDEM'] = [
            safe_extract_elevation(lambda: elev[0], 'CDEM') 
            for elev in cdem_tif.sample(coords)
        ]
        
        # Extract HRDEM values
        dataframe['HRDEM'] = [
            safe_extract_elevation(lambda: elev[0], 'HRDEM') 
            for elev in hrdem_tif.sample(coords)
        ]
        
        # Calculate combined elevation (prefer HRDEM if available)
        dataframe['pre_DGGS_Elev'] = np.where(
            np.isnan(dataframe['HRDEM']), 
            dataframe['CDEM'], 
            dataframe['HRDEM']
        )
        
        logger.info("Pre-DGGS elevation extraction completed")
        return dataframe
        
    except Exception as e:
        raise ControlPointsError(f"Failed to extract pre-DGGS elevation: {e}")


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


def extract_cdem_elevation_dggs(lon: float, lat: float, resolution: int, 
                                cdem_tif: rasterio.DatasetReader, conn: pyRserve.RConnection,
                                vertical_res: int, interp: str = 'linear') -> float:
    """
    Extract CDEM elevation using DGGS modeling.
    
    Args:
        lon, lat: Coordinates
        resolution: DGGS resolution level
        cdem_tif: CDEM raster dataset
        conn: R connection
        vertical_res: Vertical resolution for rounding
        interp: Interpolation method
        
    Returns:
        Interpolated elevation value
    """
    try:
        # Get cell centroid coordinates
        x_y = conn.eval(f'geo_to_centroid({resolution},{lon},{lat})')
        x, y = x_y[0], x_y[1]
        
        # Find neighbors and interpolate
        x_array, y_array, z_array = find_neighbors(x, y, cdem_tif)
        
        if not x_array or NO_DATA_VALUE in z_array:
            return NO_DATA_VALUE
        
        # Perform interpolation
        cdem_interp = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
        elevation = cdem_interp(x, y)[0]
        
        # Round to specified precision
        elevation = round(elevation, vertical_res)
        
        return elevation
        
    except Exception as e:
        logger.debug(f"CDEM DGGS extraction failed for ({lon}, {lat}): {e}")
        return NO_DATA_VALUE


def extract_hrdem_elevation_dggs(lon: float, lat: float, resolution: int,
                                 hrdem_tif: rasterio.DatasetReader, hrdem_extent: gpd.GeoDataFrame,
                                 conn: pyRserve.RConnection, vertical_res: int,
                                 interp: str = 'linear') -> Optional[float]:
    """
    Extract HRDEM elevation using DGGS modeling.
    
    Args:
        lon, lat: Coordinates
        resolution: DGGS resolution level
        hrdem_tif: HRDEM raster dataset
        hrdem_extent: HRDEM extent polygon
        conn: R connection
        vertical_res: Vertical resolution for rounding
        interp: Interpolation method
        
    Returns:
        Interpolated elevation value or None if not available
    """
    try:
        # Get cell centroid coordinates
        x_y = conn.eval(f'geo_to_centroid({resolution},{lon},{lat})')
        x, y = x_y[0], x_y[1]
        
        # Check if point falls within HRDEM extent
        point = shapely.geometry.Point(x, y)
        if not point.within(hrdem_extent.geometry.iloc[0]):
            return None
        
        # Find neighbors and interpolate
        x_array, y_array, z_array = find_neighbors(x, y, hrdem_tif)
        
        if not x_array or NO_DATA_VALUE in z_array:
            return None
        
        # Perform interpolation
        hrdem_interp = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
        elevation = hrdem_interp(x, y)[0]
        
        # Round to specified precision
        elevation = round(elevation, vertical_res)
        
        return elevation
        
    except Exception as e:
        logger.debug(f"HRDEM DGGS extraction failed for ({lon}, {lat}): {e}")
        return None


def process_control_points_for_resolution(control_points: pd.DataFrame, resolution: int,
                                        cdem_tif: rasterio.DatasetReader, hrdem_tif: rasterio.DatasetReader,
                                        hrdem_extent: gpd.GeoDataFrame, conn: pyRserve.RConnection) -> pd.DataFrame:
    """
    Process control points for a specific DGGS resolution.
    
    Args:
        control_points: DataFrame with control points
        resolution: DGGS resolution level
        cdem_tif: CDEM raster dataset
        hrdem_tif: HRDEM raster dataset
        hrdem_extent: HRDEM extent polygon
        conn: R connection
        
    Returns:
        DataFrame with DGGS elevation columns added
    """
    try:
        vertical_res = RESOLUTION_LOOKUP[resolution]['vertical_res']
        
        # Extract CDEM elevations
        control_points[f'model_CDEM_{resolution}'] = control_points.apply(
            lambda row: extract_cdem_elevation_dggs(
                row['lon'], row['lat'], resolution, cdem_tif, conn, vertical_res
            ), axis=1
        )
        
        # Extract HRDEM elevations
        control_points[f'model_HRDEM_{resolution}'] = control_points.apply(
            lambda row: extract_hrdem_elevation_dggs(
                row['lon'], row['lat'], resolution, hrdem_tif, hrdem_extent, conn, vertical_res
            ), axis=1
        )
        
        # Calculate combined elevation
        control_points[f'model_Elev_{resolution}'] = np.where(
            np.isnan(control_points[f'model_HRDEM_{resolution}']),
            control_points[f'model_CDEM_{resolution}'],
            control_points[f'model_HRDEM_{resolution}']
        )
        
        logger.info(f"Processed control points for resolution {resolution}")
        return control_points
        
    except Exception as e:
        raise ControlPointsError(f"Failed to process control points for resolution {resolution}: {e}")


def load_control_points() -> pd.DataFrame:
    """
    Load ground control points data.
    
    Returns:
        DataFrame with control points
    """
    try:
        control_points_path = 'Experiment_data/vege_gcp.csv'
        
        if not os.path.exists(control_points_path):
            raise ControlPointsError(f"Control points file not found: {control_points_path}")
        
        control_points = pd.read_csv(control_points_path, sep=',')
        logger.info(f"Loaded {len(control_points)} control points")
        return control_points
        
    except Exception as e:
        raise ControlPointsError(f"Failed to load control points: {e}")


def save_results(control_points: pd.DataFrame) -> None:
    """
    Save control points results to CSV file.
    
    Args:
        control_points: DataFrame with results
    """
    try:
        # Create output directory
        Path("Result").mkdir(exist_ok=True)
        
        # Save to CSV
        output_path = "Result/gcp_result.csv"
        control_points.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        raise ControlPointsError(f"Failed to save results: {e}")


def main():
    """Main function to orchestrate control points processing."""
    start_time = time.time()
    conn = None
    
    try:
        logger.info("Starting control points processing...")
        
        # Setup R connection
        conn = setup_r_connection()
        
        # Load DEM data
        cdem_tif, hrdem_tif, hrdem_extent = load_dem_data()
        
        try:
            # Load control points
            control_points = load_control_points()
            
            # Check if control points fall in HRDEM extent
            control_points['fall_in_test'] = [
                check_point_in_hrdem_extent(lon, lat, hrdem_extent)
                for lon, lat in zip(control_points.lon, control_points.lat)
            ]
            
            # Calculate pre-DGGS elevation
            control_points = extract_pre_dggs_elevation(control_points, cdem_tif, hrdem_tif)
            
            # Calculate post-DGGS elevation for each resolution level
            for resolution in range(16, 30):
                logger.info(f"Processing resolution level {resolution}...")
                control_points = process_control_points_for_resolution(
                    control_points, resolution, cdem_tif, hrdem_tif, hrdem_extent, conn
                )
            
            # Save results
            save_results(control_points)
            
            processing_time = time.time() - start_time
            logger.info(f"Control points processing completed in {processing_time:.2f} seconds")
            
        finally:
            # Clean up raster datasets
            cdem_tif.close()
            hrdem_tif.close()
            
    except ControlPointsError as e:
        logger.error(f"Control points processing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up R connection
        if conn:
            conn.shutdown()


if __name__ == "__main__":
    main()
