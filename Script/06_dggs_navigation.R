#!/usr/bin/env Rscript
#==============================================================================
# DGGS Navigation - Parent-child Look-up Tables
#==============================================================================
#
# This script generates parent-child look-up tables for DGGS navigation.
# It creates fine-resolution cells within coarse-resolution cells and
# establishes the hierarchical relationship between different resolution levels.
#
# Author: Mingke Li
# Date: 2021
#
# Usage: Rscript 06_dggs_navigation.R <dggs_resolution> <grid_id>
#==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(dggridR)
  library(dplyr)
  library(doParallel)
  library(tictoc)
  library(logging)
})

# Configure logging
basicConfig(level='INFO')
loginfo("Starting DGGS navigation script")

# Constants
V_LAT <- 37.6895
V_LON <- -51.6218
AZIMUTH <- 360 - 72.6482


#' Parse command line arguments
#' 
#' @return List containing dggs_res and fid
parse_arguments <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    stop("Usage: Rscript 06_dggs_navigation.R <dggs_resolution>")
  }
  
  dggs_res <- as.numeric(args[1])
  fid <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))
  
  # Validate resolution
  if (dggs_res < 16 || dggs_res > 29) {
    stop("Invalid DGGS resolution. Must be between 16 and 29.")
  }
  
  loginfo("DGGS resolution: %d, Grid ID: %d", dggs_res, fid)
  
  return(list(dggs_res = dggs_res, fid = fid))
}


#' Setup parallel processing
#' 
#' @return Number of cores registered
setup_parallel <- function() {
  ncores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
  registerDoParallel(cores = ncores)
  loginfo("Registered %d parallel workers", ncores)
  return(ncores)
}


#' Create DGGS objects for coarse and fine resolutions
#' 
#' @param resolution Coarse resolution level
#' @return List containing coarse and fine DGGS objects
create_dggs_objects <- function(resolution) {
  # Create coarse resolution DGGS
  dgg_coarse <- dgconstruct(
    projection = "ISEA", 
    aperture = 3, 
    topology = "HEXAGON", 
    res = resolution, 
    precision = 7, 
    azimuth_deg = AZIMUTH, 
    pole_lat_deg = V_LAT, 
    pole_lon_deg = V_LON
  )
  
  # Create fine resolution DGGS
  dgg_fine <- dgconstruct(
    projection = "ISEA", 
    aperture = 3, 
    topology = "HEXAGON", 
    res = (resolution + 1), 
    precision = 7, 
    azimuth_deg = AZIMUTH, 
    pole_lat_deg = V_LAT, 
    pole_lon_deg = V_LON
  )
  
  return(list(coarse = dgg_coarse, fine = dgg_fine))
}


#' Calculate elevation statistics for a given resolution
#' 
#' @param resolution DGGS resolution level
#' @param dataframe Data frame containing cell information
#' @param dgg_objects List containing coarse and fine DGGS objects
#' @return Data frame with parent-child relationships
elev_stats <- function(resolution, dataframe, dgg_objects) {
  tryCatch({
    # Get vertices for coarse cells
    vertices_df <- dgcellstogrid(dgg_objects$coarse, dataframe$Cell_address)
    
    # Filter to keep only the main cell (order < 7)
    vertices_df <- filter(vertices_df, order < 7)
    
    # Remove unnecessary columns
    vertices_df <- subset(vertices_df, select = -c(order, hole, piece, group))
    vertices_df <- vertices_df[, c("cell", "long", "lat")]
    
    # Rename columns in input dataframe
    dataframe <- dataframe %>% rename(cell = Cell_address, long = lon_c, lat = lat_c)
    
    # Combine vertices and centroids
    vertices_df <- rbind(vertices_df, dataframe)
    
    # Calculate fine cell addresses
    vertices_df$cell_fine <- dgGEO_to_SEQNUM(
      dgg_objects$fine, 
      vertices_df$long, 
      vertices_df$lat
    )$seqnum
    
    # Remove coordinate columns
    vertices_df <- subset(vertices_df, select = -c(long, lat))
    
    return(vertices_df)
    
  }, error = function(e) {
    logerror("Error in elev_stats for resolution %d: %s", resolution, e$message)
    return(data.frame())
  })
}


#' Load input data
#' 
#' @param dggs_res DGGS resolution
#' @param fid Grid ID
#' @return Data frame with centroid information
load_input_data <- function(dggs_res, fid) {
  input_file <- sprintf("Result/Level%d/Centroid/vege_pre_%d.csv", dggs_res, fid)
  
  if (!file.exists(input_file)) {
    stop("Input file not found: ", input_file)
  }
  
  tryCatch({
    coarse_df <- read.csv(input_file, header = TRUE)[, c("Cell_address", "lon_c", "lat_c")]
    loginfo("Loaded %d rows from %s", nrow(coarse_df), input_file)
    return(coarse_df)
    
  }, error = function(e) {
    stop("Failed to load input data: ", e$message)
  })
}


#' Split dataframe for parallel processing
#' 
#' @param dataframe Input data frame
#' @param ncores Number of cores
#' @return List of dataframe chunks
split_dataframe <- function(dataframe, ncores) {
  if (ncores <= 1) {
    return(list(dataframe))
  }
  
  # Split dataframe randomly
  split_indices <- sample(1:ncores, nrow(dataframe), replace = TRUE)
  df_split <- split(dataframe, split_indices)
  
  # Remove empty chunks
  df_split <- df_split[sapply(df_split, nrow) > 0]
  
  loginfo("Split dataframe into %d chunks", length(df_split))
  return(df_split)
}


#' Save results to CSV file
#' 
#' @param output_df Data frame to save
#' @param dggs_res DGGS resolution
#' @param fid Grid ID
save_results <- function(output_df, dggs_res, fid) {
  # Create output directory
  output_dir <- sprintf("Result/Level%d/Temp", dggs_res)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save to CSV
  output_file <- sprintf("%s/vege_temp_%d.csv", output_dir, fid)
  write.csv(output_df, output_file, row.names = FALSE)
  
  loginfo("Results saved to %s", output_file)
}


#' Main function
main <- function() {
  tryCatch({
    # Parse arguments
    args <- parse_arguments()
    dggs_res <- args$dggs_res
    fid <- args$fid
    
    # Setup parallel processing
    ncores <- setup_parallel()
    
    # Load input data
    coarse_df <- load_input_data(dggs_res, fid)
    
    # Create DGGS objects
    dgg_objects <- create_dggs_objects(dggs_res)
    
    # Split dataframe for parallel processing
    coarse_df_split <- split_dataframe(coarse_df, ncores)
    
    # Clean up original dataframe
    rm(coarse_df)
    
    # Record processing time
    tic("Generate parent-child relationships")
    
    # Parallel processing
    output_df <- foreach(
      df = coarse_df_split, 
      .combine = rbind,
      .packages = 'dggridR'
    ) %dopar% {
      elev_stats(dggs_res, df, dgg_objects)
    }
    
    # Record processing time
    toc()
    
    # Save results
    save_results(output_df, dggs_res, fid)
    
    loginfo("DGGS navigation completed successfully")
    
  }, error = function(e) {
    logerror("Script failed: %s", e$message)
    quit(status = 1)
  })
}


# Run main function if script is executed directly
if (!interactive()) {
  main()
}
