#!/usr/bin/env python3
"""
CDEM Pre-processing Script for DGGS Elevation Integration

This script handles the pre-processing of Canadian Digital Elevation Model (CDEM) data:
- Mosaicking multiple CDEM tiles
- Converting to NAD83 CSRS geographic coordinate system
- Converting vertical datum from CGVD1928 to CGVD2013
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
from rasterio.warp import reproject, Resampling
import gdal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cdem_preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NAD83_CSRS_EPSG = "EPSG:4617"
NO_DATA_VALUE = -32767.0
COMPRESSION = 'lzw'


class CDEMPreprocessError(Exception):
    """Custom exception for CDEM pre-processing errors."""
    pass


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    Path("Data").mkdir(exist_ok=True)
    Path("Result").mkdir(exist_ok=True)


def find_cdem_files() -> List[str]:
    """
    Find all CDEM files in the Data directory.
    
    Returns:
        List of CDEM file paths
    """
    cdem_files = glob.glob('Data/CDEM_*.tif')
    if not cdem_files:
        raise CDEMPreprocessError("No CDEM files found in Data directory")
    
    logger.info(f"Found {len(cdem_files)} CDEM files")
    return cdem_files


def mosaic_cdem_files(cdem_files: List[str]) -> str:
    """
    Mosaic multiple CDEM files into a single file.
    
    Args:
        cdem_files: List of CDEM file paths
        
    Returns:
        Path to the mosaicked file
    """
    logger.info("Starting CDEM mosaicking...")
    
    if len(cdem_files) == 1:
        # If only one file, just rename it
        src = rasterio.open(cdem_files[0])
        src.close()
        output_path = 'Data/CDEM_mosaic.tif'
        os.rename(cdem_files[0], output_path)
        logger.info("Single CDEM file renamed to mosaic")
        return output_path
    
    # Open all source files
    src_files = []
    try:
        for file_path in cdem_files:
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
            "nodata": NO_DATA_VALUE,
            "transform": mosaic_trans,
            "crs": NAD83_CSRS_EPSG
        })
        
        # Write mosaic to disk
        output_path = "Data/CDEM_mosaic.tif"
        with rasterio.open(output_path, "w", **mosaic_meta) as dest:
            dest.write(mosaic_dem)
        
        logger.info(f"CDEM mosaic created successfully: {output_path}")
        return output_path
        
    finally:
        # Clean up
        for src in src_files:
            src.close()
        del mosaic_dem, mosaic_trans, mosaic_meta
        gc.collect()


def reproject_to_geographic(input_path: str, output_path: str) -> None:
    """
    Reproject CDEM to NAD83 CSRS geographic coordinate system.
    
    Args:
        input_path: Path to input raster
        output_path: Path to output raster
    """
    logger.info("Reprojecting CDEM to geographic coordinates...")
    
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
        raise CDEMPreprocessError(f"Failed to reproject CDEM: {e}")


def get_features_from_geodataframe(gdf: gpd.GeoDataFrame) -> List[dict]:
    """
    Parse features from GeoDataFrame in a format that rasterio expects.
    
    Args:
        gdf: GeoDataFrame
        
    Returns:
        List of feature geometries
    """
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def crop_raster_to_study_area(raster_path: str, study_area_path: str, output_path: str) -> None:
    """
    Crop raster to the extent of study area.
    
    Args:
        raster_path: Path to input raster
        study_area_path: Path to study area shapefile
        output_path: Path to output raster
    """
    logger.info("Cropping raster to study area...")
    
    try:
        # Load raster and study area
        with rasterio.open(raster_path) as raster:
            study_area = gpd.GeoDataFrame.from_file(study_area_path)
            study_area = study_area.set_crs(NAD83_CSRS_EPSG)
            
            # Get features for masking
            features = get_features_from_geodataframe(study_area)
            
            # Perform masking
            out_image, out_transform = mask(
                dataset=raster, 
                shapes=features, 
                crop=True
            )
            
            # Update metadata
            out_meta = raster.meta.copy()
            out_meta.update({
                'height': int(out_image.shape[1]),
                'width': int(out_image.shape[2]),
                'transform': out_transform,
                'compress': COMPRESSION
            })
            
            # Write output
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        logger.info(f"Raster cropped successfully: {output_path}")
        
    except Exception as e:
        raise CDEMPreprocessError(f"Failed to crop raster: {e}")


def reproject_image_to_master(master_path: str, slave_path: str) -> str:
    """
    Reproject a raster (slave) to match the extent, resolution and projection of another raster (master).
    
    Args:
        master_path: Path to master raster
        slave_path: Path to slave raster
        
    Returns:
        Path to reprojected slave raster
    """
    logger.info("Reprojecting slave raster to match master...")
    
    try:
        # Open datasets
        slave_ds = gdal.Open(slave_path)
        master_ds = gdal.Open(master_path)
        
        if not slave_ds or not master_ds:
            raise CDEMPreprocessError("Failed to open raster datasets")
        
        # Get properties
        slave_proj = slave_ds.GetProjection()
        data_type = slave_ds.GetRasterBand(1).DataType
        n_bands = slave_ds.RasterCount
        master_proj = master_ds.GetProjection()
        master_geotrans = master_ds.GetGeoTransform()
        w = master_ds.RasterXSize
        h = master_ds.RasterYSize
        
        # Create output filename
        output_path = slave_path.replace(".byn", "_temp.tif")
        
        # Create output dataset
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            output_path, w, h, n_bands, data_type
        )
        dst_ds.SetGeoTransform(master_geotrans)
        dst_ds.SetProjection(master_proj)
        dst_ds.GetRasterBand(1).SetNoDataValue(NO_DATA_VALUE)
        
        # Perform reprojection
        gdal.ReprojectImage(
            slave_ds, dst_ds, slave_proj, master_proj, 
            gdal.GRA_NearestNeighbour
        )
        
        # Clean up
        dst_ds = None
        slave_ds = None
        master_ds = None
        
        logger.info(f"Reprojection completed: {output_path}")
        return output_path
        
    except Exception as e:
        raise CDEMPreprocessError(f"Failed to reproject image: {e}")


def convert_to_cgvd2013(cdem_path: str, byn_path: str, output_path: str) -> None:
    """
    Convert vertical datum from CGVD1928 to CGVD2013.
    
    Args:
        cdem_path: Path to CDEM raster (CGVD1928)
        byn_path: Path to BYN raster (conversion grid)
        output_path: Path to output raster (CGVD2013)
    """
    logger.info("Converting vertical datum to CGVD2013...")
    
    try:
        # Load rasters
        with rasterio.open(cdem_path) as cdem_src, rasterio.open(byn_path) as byn_src:
            # Read data
            cgvd1928 = cdem_src.read(1)
            delta_dem = byn_src.read(1)
            
            # Perform conversion: CGVD2013 = CGVD1928 - delta_dem/1000
            cgvd2013 = cgvd1928 - delta_dem / 1000
            
            # Handle invalid values
            cgvd2013 = np.where(cgvd2013 < -1000, NO_DATA_VALUE, cgvd2013)
            
            # Update metadata
            cgvd2013_meta = cdem_src.meta.copy()
            cgvd2013_meta.update({"compress": COMPRESSION})
            
            # Write output
            with rasterio.open(output_path, "w", **cgvd2013_meta) as dest:
                dest.write(cgvd2013.astype(rasterio.float32), 1)
        
        logger.info(f"Vertical datum conversion completed: {output_path}")
        
    except Exception as e:
        raise CDEMPreprocessError(f"Failed to convert vertical datum: {e}")


def main():
    """Main function to orchestrate CDEM pre-processing."""
    try:
        logger.info("Starting CDEM pre-processing...")
        
        # Setup directories
        setup_directories()
        
        # Step 1: Mosaic CDEM files
        cdem_files = find_cdem_files()
        mosaic_path = mosaic_cdem_files(cdem_files)
        
        # Step 2: Reproject to geographic coordinates
        reprojected_path = 'Data/CDEM_mosaic_4617.tif'
        reproject_to_geographic(mosaic_path, reprojected_path)
        
        # Step 3: Crop to study area
        cropped_path = 'Data/CDEM_mosaic_temp.tif'
        crop_raster_to_study_area(
            reprojected_path, 
            'Data/study_area.shp', 
            cropped_path
        )
        
        # Step 4: Unify CDEM and BYN rasters
        byn_temp_path = reproject_image_to_master(
            cropped_path, 
            'Data/HT2_2010v70_CGG2013a.byn'
        )
        
        # Step 5: Convert to CGVD2013
        convert_to_cgvd2013(
            cropped_path, 
            byn_temp_path, 
            'Data/CDEM_cgvd2013.tif'
        )
        
        logger.info("CDEM pre-processing completed successfully")
        
    except CDEMPreprocessError as e:
        logger.error(f"CDEM pre-processing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
