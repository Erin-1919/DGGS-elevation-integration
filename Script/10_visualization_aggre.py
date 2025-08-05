#!/usr/bin/env python3
"""
Aggregation Visualization Script for DGGS Elevation Integration

This script creates visualizations of aggregation results:
- Loads centroid and statistics data
- Creates raster-style visualizations for mean, max, and min statistics
- Saves images for each grid cell

Author: Mingke Li
Date: 2021
"""

import sys
import os
import logging
import gc
from pathlib import Path
from typing import List, Tuple, Optional

import datashader as ds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from datashader.mpl_ext import dsshow

# Configure matplotlib for non-interactive backend
matplotlib.use('AGG')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization_aggre.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


def validate_arguments() -> Tuple[int, int]:
    """
    Validate and parse command line arguments.
    
    Returns:
        Tuple containing (dggs_res, grid_num)
        
    Raises:
        VisualizationError: If arguments are invalid
    """
    if len(sys.argv) < 3:
        raise VisualizationError("Usage: python 10_visualization_aggre.py <dggs_resolution> <grid_num>")
    
    try:
        dggs_res = int(sys.argv[1])
        grid_num = int(sys.argv[2])
    except (ValueError, TypeError) as e:
        raise VisualizationError(f"Invalid argument format: {e}")
    
    if dggs_res < 16 or dggs_res > 29:
        raise VisualizationError(f"Invalid DGGS resolution: {dggs_res}")
    
    if grid_num < 1:
        raise VisualizationError(f"Invalid grid number: {grid_num}")
    
    return dggs_res, grid_num


def load_data_for_grid(dggs_res: int, fid: int) -> pd.DataFrame:
    """
    Load centroid and statistics data for a specific grid.
    
    Args:
        dggs_res: DGGS resolution level
        fid: Grid ID
        
    Returns:
        DataFrame with merged data
    """
    try:
        # Load centroid data
        centroid_path = f'Result/Level{dggs_res}/Centroid/vege_pre_{fid}.csv'
        if not os.path.exists(centroid_path):
            raise VisualizationError(f"Centroid file not found: {centroid_path}")
        
        centroid_df = pd.read_csv(centroid_path)
        logger.info(f"Loaded centroid data: {len(centroid_df)} rows")
        
        # Load statistics data
        stats_path = f'Result/Level{dggs_res}/Stats/vege_temp_{fid}.csv'
        if not os.path.exists(stats_path):
            raise VisualizationError(f"Statistics file not found: {stats_path}")
        
        stats_df = pd.read_csv(stats_path)
        logger.info(f"Loaded statistics data: {len(stats_df)} rows")
        
        # Merge dataframes
        merge_df = pd.merge(
            left=stats_df, 
            right=centroid_df, 
            how="inner", 
            on="Cell_address"
        )
        
        # Clean up memory
        del centroid_df, stats_df
        gc.collect()
        
        logger.info(f"Merged data: {len(merge_df)} rows")
        return merge_df
        
    except Exception as e:
        raise VisualizationError(f"Failed to load data for grid {fid}: {e}")


def create_statistic_visualization(dataframe: pd.DataFrame, dggs_res: int, fid: int, 
                                  statistic: str, aggregator_func) -> None:
    """
    Create and save visualization for a specific statistic.
    
    Args:
        dataframe: DataFrame with data to visualize
        dggs_res: DGGS resolution level
        fid: Grid ID
        statistic: Name of the statistic ('mean', 'max', 'min')
        aggregator_func: Datashader aggregator function
    """
    try:
        # Create output directory
        output_dir = Path(f"Result/Level{dggs_res}/Img")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Create datashader visualization
        artist = dsshow(
            dataframe, 
            ds.Point('lon_c', 'lat_c'), 
            aggregator=aggregator_func(f'elev_{statistic}'), 
            cmap='gray', 
            vmin=0, 
            vmax=600, 
            plot_width=500, 
            plot_height=500, 
            ax=ax
        )
        
        # Remove axis labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        
        # Save figure
        output_path = output_dir / f'vege_{statistic}_{fid}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # Clean up memory
        gc.collect()
        
        logger.info(f"{statistic.capitalize()} visualization saved: {output_path}")
        
    except Exception as e:
        raise VisualizationError(f"Failed to create {statistic} visualization for grid {fid}: {e}")


def create_all_statistic_visualizations(dataframe: pd.DataFrame, dggs_res: int, fid: int) -> None:
    """
    Create visualizations for all statistics (mean, max, min).
    
    Args:
        dataframe: DataFrame with data to visualize
        dggs_res: DGGS resolution level
        fid: Grid ID
    """
    try:
        # Create mean visualization
        create_statistic_visualization(dataframe, dggs_res, fid, 'mean', ds.mean)
        
        # Create max visualization
        create_statistic_visualization(dataframe, dggs_res, fid, 'max', ds.max)
        
        # Create min visualization
        create_statistic_visualization(dataframe, dggs_res, fid, 'min', ds.min)
        
        logger.info(f"All statistic visualizations created for grid {fid}")
        
    except Exception as e:
        raise VisualizationError(f"Failed to create all statistic visualizations for grid {fid}: {e}")


def process_all_grids(dggs_res: int, grid_num: int) -> None:
    """
    Process all grids for visualization.
    
    Args:
        dggs_res: DGGS resolution level
        grid_num: Number of grids to process
    """
    logger.info(f"Processing {grid_num} grids for resolution {dggs_res}")
    
    for fid in range(1, grid_num + 1):
        try:
            logger.info(f"Processing grid {fid}/{grid_num}")
            
            # Load data
            merge_df = load_data_for_grid(dggs_res, fid)
            
            # Create all statistic visualizations
            create_all_statistic_visualizations(merge_df, dggs_res, fid)
            
        except VisualizationError as e:
            logger.error(f"Failed to process grid {fid}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing grid {fid}: {e}")
            continue


def main():
    """Main function to orchestrate aggregation visualization."""
    try:
        # Validate arguments
        dggs_res, grid_num = validate_arguments()
        logger.info(f"Starting aggregation visualization for resolution {dggs_res}, {grid_num} grids")
        
        # Process all grids
        process_all_grids(dggs_res, grid_num)
        
        logger.info("Aggregation visualization completed successfully")
        
    except VisualizationError as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
        
