#!/usr/bin/env python3
"""
DGGS Aggregation Script for Elevation Integration

This script performs elevation generalization by calculating multiple statistics
on elevation data across different DGGS resolution levels. It aggregates
fine-resolution data to coarse-resolution cells.

Author: Mingke Li
Date: 2021
"""

import sys
import os
import time
import logging
import gc
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import multiprocess as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dggs_aggregation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NO_DATA_VALUE = -32767.0

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


class DGGSAggregationError(Exception):
    """Custom exception for DGGS aggregation errors."""
    pass


def validate_arguments() -> Tuple[int, int]:
    """
    Validate and parse command line arguments.
    
    Returns:
        Tuple containing (dggs_res, fid)
        
    Raises:
        DGGSAggregationError: If arguments are invalid
    """
    if len(sys.argv) < 2:
        raise DGGSAggregationError("Usage: python 07_dggs_aggregation.py <dggs_resolution>")
    
    try:
        dggs_res = int(sys.argv[1])
        fid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    except (ValueError, TypeError) as e:
        raise DGGSAggregationError(f"Invalid argument format: {e}")
    
    if dggs_res not in RESOLUTION_LOOKUP:
        raise DGGSAggregationError(f"Invalid DGGS resolution: {dggs_res}")
    
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
        raise DGGSAggregationError(f"Invalid DGGS resolution: {dggs_res}")
    
    return params['cell_size'], params['vertical_res']


def load_input_data(dggs_res: int, fid: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load input data for aggregation.
    
    Args:
        dggs_res: DGGS resolution level
        fid: Grid ID
        
    Returns:
        Tuple containing (cell_df, elev_df)
    """
    try:
        # Load cell data
        cell_csv_path = f"Result/Level{dggs_res}/Temp/vege_temp_{fid}.csv"
        if not os.path.exists(cell_csv_path):
            raise DGGSAggregationError(f"Cell file not found: {cell_csv_path}")
        
        cell_df = pd.read_csv(cell_csv_path, index_col='cell_fine')
        logger.info(f"Loaded cell data: {len(cell_df)} rows")
        
        # Load elevation data
        if dggs_res == 28:
            elev_csv_path = f"Result/Level{dggs_res+1}/Elev/vege_temp_{fid}.csv"
        else:
            elev_csv_path = f"Result/Level{dggs_res+1}/Stats/vege_temp_{fid}.csv"
        
        if not os.path.exists(elev_csv_path):
            raise DGGSAggregationError(f"Elevation file not found: {elev_csv_path}")
        
        elev_df = pd.read_csv(elev_csv_path, index_col='Cell_address')
        logger.info(f"Loaded elevation data: {len(elev_df)} rows")
        
        return cell_df, elev_df
        
    except Exception as e:
        raise DGGSAggregationError(f"Failed to load input data: {e}")


def join_dataframes(cell_df: pd.DataFrame, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join cell and elevation dataframes.
    
    Args:
        cell_df: Cell dataframe
        elev_df: Elevation dataframe
        
    Returns:
        Joined dataframe
    """
    try:
        # Join dataframes
        join_df = cell_df.join(elev_df, how='left')
        
        # Clean up memory
        del cell_df, elev_df
        gc.collect()
        
        logger.info(f"Joined dataframes: {len(join_df)} rows")
        return join_df
        
    except Exception as e:
        raise DGGSAggregationError(f"Failed to join dataframes: {e}")


def calculate_elevation_statistics(dataframe: pd.DataFrame, dggs_res: int, 
                                 vertical_res: int) -> pd.DataFrame:
    """
    Calculate elevation statistics for each cell.
    
    Args:
        dataframe: Input dataframe
        dggs_res: DGGS resolution level
        vertical_res: Vertical resolution for rounding
        
    Returns:
        DataFrame with elevation statistics
    """
    try:
        if dggs_res == 28:
            # For finest resolution, combine HRDEM and CDEM data
            dataframe['elev_stats'] = np.where(
                np.isnan(dataframe['model_hrdem']), 
                dataframe['model_cdem'], 
                dataframe['model_hrdem']
            )
            dataframe = dataframe.drop(columns=['model_hrdem', 'model_cdem'])
            
            # Calculate statistics
            join_stats_df = dataframe.groupby(["cell"]).agg({
                'elev_stats': ['mean', 'max', 'min']
            })
            
            # Flatten column names
            join_stats_df.columns = ['elev_mean', 'elev_max', 'elev_min']
            
        else:
            # For other resolutions, aggregate existing statistics
            join_stats_df = dataframe.groupby(["cell"]).agg({
                'elev_mean': 'mean',
                'elev_max': 'max', 
                'elev_min': 'min'
            })
        
        # Round to specified precision
        for col in ['elev_mean', 'elev_max', 'elev_min']:
            if col in join_stats_df.columns:
                join_stats_df[col] = [round(row, vertical_res) for row in join_stats_df[col]]
        
        # Set index name
        join_stats_df.index.names = ['Cell_address']
        
        return join_stats_df
        
    except Exception as e:
        raise DGGSAggregationError(f"Failed to calculate elevation statistics: {e}")


def process_dataframe_chunk(chunk: pd.DataFrame, dggs_res: int, 
                           vertical_res: int) -> pd.DataFrame:
    """
    Process a chunk of the dataframe to calculate statistics.
    
    Args:
        chunk: DataFrame chunk to process
        dggs_res: DGGS resolution level
        vertical_res: Vertical resolution for rounding
        
    Returns:
        DataFrame with elevation statistics
    """
    try:
        return calculate_elevation_statistics(chunk, dggs_res, vertical_res)
    except Exception as e:
        logger.error(f"Failed to process dataframe chunk: {e}")
        raise


def split_dataframe_for_parallel(dataframe: pd.DataFrame, n_cores: int) -> List[pd.DataFrame]:
    """
    Split dataframe for parallel processing.
    
    Args:
        dataframe: Input dataframe
        n_cores: Number of cores
        
    Returns:
        List of dataframe chunks
    """
    try:
        # Get value counts for balanced splitting
        vc = dataframe.cell.value_counts().rename('cnt')
        vc = vc.to_frame().assign(bin=[i % n_cores for i in range(vc.size)])
        
        # Create chunks based on cell values
        df_split = []
        for i in range(n_cores):
            cells_in_bin = vc[vc.bin == i].index
            chunk = dataframe[dataframe.cell.isin(cells_in_bin)]
            if not chunk.empty:
                df_split.append(chunk)
        
        logger.info(f"Split dataframe into {len(df_split)} chunks")
        return df_split
        
    except Exception as e:
        raise DGGSAggregationError(f"Failed to split dataframe: {e}")


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
        output_dir = Path(f"Result/Level{dggs_res}/Stats")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = output_dir / f'vege_temp_{fid}.csv'
        df.to_csv(output_path)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        raise DGGSAggregationError(f"Failed to save results: {e}")


def main():
    """Main function to orchestrate DGGS aggregation."""
    start_time = time.time()
    
    try:
        # Validate arguments
        dggs_res, fid = validate_arguments()
        logger.info(f"Starting DGGS aggregation for resolution {dggs_res}, grid {fid}")
        
        # Get resolution parameters
        cell_size, vertical_res = get_resolution_params(dggs_res)
        logger.info(f"Cell size: {cell_size}, Vertical resolution: {vertical_res}")
        
        # Load input data
        cell_df, elev_df = load_input_data(dggs_res, fid)
        
        # Join dataframes
        join_df = join_dataframes(cell_df, elev_df)
        
        # Setup parallel processing
        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        logger.info(f"Using {n_cores} cores for parallel processing")
        
        # Split dataframe for parallel processing
        df_split = split_dataframe_for_parallel(join_df, n_cores)
        
        # Process in parallel
        with mp.Pool(processes=n_cores) as pool:
            results = pool.map(
                lambda chunk: process_dataframe_chunk(chunk, dggs_res, vertical_res),
                df_split
            )
        
        # Combine results
        df_output = pd.concat(results)
        
        # Save results
        save_results(df_output, dggs_res, fid)
        
        processing_time = time.time() - start_time
        logger.info(f"DGGS aggregation completed in {processing_time:.2f} seconds")
        
    except DGGSAggregationError as e:
        logger.error(f"DGGS aggregation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
