#!/usr/bin/env python3
"""
HRDEM Pre-processing Script for DGGS Elevation Integration

This script handles the pre-processing of High Resolution Digital Elevation Model (HRDEM) data:
- Reprojecting HRDEM to geographic coordinate system
- Mosaicking multiple HRDEM tiles
- Cropping to study area extent

Author: Mingke Li
Date: 2021
"""

import sys
import os
import gc
import glob
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.merge import merge
from rasterio.mask import mask
import gdal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hrdem_preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NAD83_CSRS_EPSG = "EPSG:4617"
NO_DATA_VALUE = -32767.0
COMPRESSION = 'lzw'


class HRDEMPreprocessError(Exception):
    """Custom exception for HRDEM pre-processing errors."""
    pass


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    Path("Data").mkdir(exist_ok=True)
    Path("Result").mkdir(exist_ok=True)


def find_hrdem_files() -> List[str]:
    """
    Find all HRDEM files in the Data directory.
    
    Returns:
        List of HRDEM file paths
    """
    hrdem_files = glob.glob('Data/HRDEM_*.tif')
    if not hrdem_files:
        raise HRDEMPreprocessError("No HRDEM files found in Data directory")
    
    logger.info(f"Found {len(hrdem_files)} HRDEM files")
    return hrdem_files


def reproject_hrdem_to_geographic(input_path: str, output_path: str) -> None:
    """
    Reproject HRDEM to NAD83 CSRS geographic coordinate system.
    
    Args:
        input_path: Path to input HRDEM raster
        output_path: Path to output raster
    """
    logger.info(f"Reprojecting {input_path} to geographic coordinates...")
    
    try:
        with rasterio.open(input_path) as src:
            # Define warp options
            warp_options = gdal.WarpOptions(
                format=src.meta.get('driver'),
                outputType=gdal.GDT_Float32,
                srcSRS=src.meta.get('crs'),
                dstSRS=NAD83_CSRS_EPSG,
                dstNodata=src.meta.get('nodata'),
                creationOptions=[f'COMPRESS={COMPRESSION.upper()}']
            )
            
            # Perform warp
            gdal.Warp(output_path, input_path, options=warp_options)
        
        logger.info(f"Reprojection completed: {output_path}")
        
    except Exception as e:
        raise HRDEMPreprocessError(f"Failed to reproject HRDEM {input_path}: {e}")


def reproject_all_hrdem_files(hrdem_files: List[str]) -> List[str]:
    """
    Reproject all HRDEM files to geographic coordinate system.
    
    Args:
        hrdem_files: List of HRDEM file paths
        
    Returns:
        List of reprojected file paths
    """
    logger.info("Reprojecting all HRDEM files...")
    
    reprojected_files = []
    for file_path in hrdem_files:
        output_path = file_path[:-4] + '_4617.tif'
        reproject_hrdem_to_geographic(file_path, output_path)
        reprojected_files.append(output_path)
    
    logger.info("All HRDEM files reprojected successfully")
    return reprojected_files


def mosaic_hrdem_files(hrdem_files: List[str]) -> str:
    """
    Mosaic multiple HRDEM files into a single file.
    
    Args:
        hrdem_files: List of HRDEM file paths
        
    Returns:
        Path to the mosaicked file
    """
    logger.info("Starting HRDEM mosaicking...")
    
    if len(hrdem_files) == 1:
        # If only one file, just rename it
        src = rasterio.open(hrdem_files[0])
        src.close()
        output_path = 'Data/HRDEM_mosaic.tif'
        os.rename(hrdem_files[0], output_path)
        logger.info("Single HRDEM file renamed to mosaic")
        return output_path
    
    # Open all source files
    src_files = []
    try:
        for file_path in hrdem_files:
            src = rasterio.open(file_path)
            src_files.append(src)
        
        # Merge files
        mosaic_dem, mosaic_trans = merge(
            src_files, 
            res=src_files[0].res, 
            nodata=NO_DATA_VALUE, 
            method='first'
        )
        
        # Copy metadata and update
        mosaic_meta = src_files[0].meta.copy()
        mosaic_meta.update({
            "driver": "GTiff",
            "height": mosaic_dem.shape[1],
            "width": mosaic_dem.shape[2],
            "compress": COMPRESSION,
            "count": 1,
            "dtype": 'float32',
            "nodata": NO_DATA_VALUE,
            "transform": mosaic_trans,
            "crs": NAD83_CSRS_EPSG
        })
        
        # Write mosaic to disk
        output_path = "Data/HRDEM_mosaic.tif"
        with rasterio.open(output_path, "w", **mosaic_meta) as dest:
            dest.write(mosaic_dem)
        
        logger.info(f"HRDEM mosaic created successfully: {output_path}")
        return output_path
        
    finally:
        # Clean up
        for src in src_files:
            src.close()
        del mosaic_dem, mosaic_trans, mosaic_meta
        gc.collect()


def get_features_from_geodataframe(gdf: gpd.GeoDataFrame) -> List[dict]:
    """
    Parse features from GeoDataFrame in a format that rasterio expects.
    
    Args:
        gdf: GeoDataFrame
        
    Returns:
        List of feature geometries
    """
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def crop_hrdem_to_study_area(hrdem_path: str, study_area_path: str, output_path: str) -> None:
    """
    Crop HRDEM to the extent of study area.
    
    Args:
        hrdem_path: Path to HRDEM raster
        study_area_path: Path to study area shapefile
        output_path: Path to output raster
    """
    logger.info("Cropping HRDEM to study area...")
    
    try:
        # Load raster and study area
        with rasterio.open(hrdem_path) as hrdem:
            study_area = gpd.GeoDataFrame.from_file(study_area_path)
            study_area = study_area.set_crs(NAD83_CSRS_EPSG)
            
            # Get features for masking
            features = get_features_from_geodataframe(study_area)
            
            # Perform masking
            out_image, out_transform = mask(
                dataset=hrdem, 
                shapes=features, 
                crop=True
            )
            
            # Update metadata
            out_meta = hrdem.meta.copy()
            out_meta.update({
                'height': int(out_image.shape[1]),
                'width': int(out_image.shape[2]),
                'transform': out_transform,
                'compress': COMPRESSION
            })
            
            # Write output
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        logger.info(f"HRDEM cropped successfully: {output_path}")
        
    except Exception as e:
        raise HRDEMPreprocessError(f"Failed to crop HRDEM: {e}")


def main():
    """Main function to orchestrate HRDEM pre-processing."""
    try:
        logger.info("Starting HRDEM pre-processing...")
        
        # Setup directories
        setup_directories()
        
        # Step 1: Find HRDEM files
        hrdem_files = find_hrdem_files()
        
        # Step 2: Reproject all HRDEM files to geographic coordinates
        reprojected_files = reproject_all_hrdem_files(hrdem_files)
        
        # Step 3: Mosaic HRDEM files
        mosaic_path = mosaic_hrdem_files(reprojected_files)
        
        # Step 4: Crop to study area
        crop_hrdem_to_study_area(
            mosaic_path, 
            'Data/study_area.shp', 
            'Data/HRDEM_cgvd2013.tif'
        )
        
        logger.info("HRDEM pre-processing completed successfully")
        
    except HRDEMPreprocessError as e:
        logger.error(f"HRDEM pre-processing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
