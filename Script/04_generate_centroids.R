#!/usr/bin/env Rscript
#==============================================================================
# Generate Centroids within Area of Interest
#==============================================================================
#
# This script generates DGGS cell centroids within the area of interest
# for parallel processing. It uses the dggridR library to create ISEA3H
# hexagonal grid cells and extracts their centroids.
#
# Author: Mingke Li
# Date: 2021
#
# Usage: Rscript 04_generate_centroids.R <dggs_resolution> <grid_id>
#==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(dggridR)
  library(rgdal)
  library(rgeos)
  library(dplyr)
  library(doParallel)
  library(tictoc)
  library(logging)
})

# Configure logging
basicConfig(level='INFO')
loginfo("Starting centroid generation script")

# Constants
V_LAT <- 37.6895
V_LON <- -51.6218
AZIMUTH <- 360 - 72.6482

# Resolution lookup table
RESOLUTION_LOOKUP <- data.frame(
  res_list = c(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
  cell_size_list = c(0.005, 0.003, 0.001, 0.0009, 0.0008, 0.0003, 0.0003, 
                     0.0001, 0.0001, 0.00006, 0.00003, 0.00002, 0.00001, 0.000005),
  vertical_res_list = c(0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6)
)


#' Parse command line arguments
#' 
#' @return List containing dggs_res and fid
parse_arguments <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    stop("Usage: Rscript 04_generate_centroids.R <dggs_resolution>")
  }
  
  dggs_res <- as.numeric(args[1])
  fid <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))
  
  # Validate resolution
  if (!dggs_res %in% RESOLUTION_LOOKUP$res_list) {
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


#' Get cell size and vertical resolution for given DGGS resolution
#' 
#' @param dggs_res DGGS resolution level
#' @return List containing cell_size and vertical_res
get_resolution_params <- function(dggs_res) {
  row <- RESOLUTION_LOOKUP[RESOLUTION_LOOKUP$res_list == dggs_res, ]
  
  if (nrow(row) == 0) {
    stop("Invalid DGGS resolution: ", dggs_res)
  }
  
  return(list(
    cell_size = row$cell_size_list,
    vertical_res = row$vertical_res_list
  ))
}


#' Create DGGS object with optimal orientation for Canada
#' 
#' @param resolution DGGS resolution level
#' @return DGGS object
create_dggs <- function(resolution) {
  # Reference: Zhou, J., Ben, J., Wang, R., Zheng, M., Yao, X., & Du, L. (2020). 
  # A novel method of determining the optimal polyhedral orientation for discrete 
  # global grid systems applicable to regional-scale areas of interest. 
  # International Journal of Digital Earth, 13(12), 1553-1569.
  
  dgg <- dgconstruct(
    projection = "ISEA", 
    aperture = 3, 
    topology = "HEXAGON", 
    res = resolution, 
    azimuth_deg = AZIMUTH, 
    pole_lat_deg = V_LAT, 
    pole_lon_deg = V_LON
  )
  
  return(dgg)
}


#' Generate nested grids for parallel processing
#' 
#' @param sqnum Number of grid cells (should be square of number of cores)
#' @param fid Grid ID for this parallel task
#' @return SpatialPolygons object containing the grid
nested_grids <- function(sqnum, fid) {
  loginfo("Generating nested grids for fid %d", fid)
  
  # Read fishnet grid
  fishnet_all <- readOGR(dsn = "Data", layer = 'fishnet_grid')
  
  if (fid > length(fishnet_all)) {
    stop("Grid ID ", fid, " exceeds number of available grids (", length(fishnet_all), ")")
  }
  
  # Get bounding box for this grid
  fishnet <- bbox(fishnet_all[fid, ])
  minx <- fishnet[1, 1]
  miny <- fishnet[2, 1]
  maxx <- fishnet[1, 2]
  maxy <- fishnet[2, 2]
  
  # Calculate grid dimensions
  width <- max((maxx - minx), (maxy - miny)) / (2 * sqrt(sqnum))
  
  # Create polygon coordinates
  coords <- matrix(
    c(minx, miny, maxx, miny, maxx, maxy, maxx, miny, minx, miny), 
    byrow = TRUE, 
    ncol = 2
  )
  
  # Create spatial polygon
  regbox <- Polygon(coords)
  regbox <- SpatialPolygons(
    list(Polygons(list(regbox), ID = "a")), 
    proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  )
  
  # Generate grid points
  reg_points <- sp::makegrid(regbox, n = sqnum)
  reg_points <- sp::SpatialPoints(cbind(reg_points$x1, reg_points$x2))
  
  # Create buffer around points
  reg_grid <- rgeos::gBuffer(reg_points, byid = TRUE, width = width, capStyle = "SQUARE")
  
  return(reg_grid)
}


#' Find cell centroids within a rectangular searching area
#' 
#' @param i Grid index
#' @param cellsize Cell size for the DGGS resolution
#' @param dgg DGGS object
#' @param rectangles SpatialPolygons object containing the grid
#' @return Data frame with centroid information
find_centroid <- function(i, cellsize, dgg, rectangles) {
  tryCatch({
    # Generate centroids
    centroids <- sp::makegrid(rectangles[i], cellsize = cellsize)
    
    # Convert to DGGS cell addresses
    centroids$Cell_address <- dgGEO_to_SEQNUM(dgg, centroids$x1, centroids$x2)$seqnum
    
    # Remove duplicates
    centroids <- centroids[!duplicated(centroids$Cell_address), ]
    
    # Get centroid coordinates
    centroid_coords <- dgSEQNUM_to_GEO(dgg, centroids$Cell_address)
    centroids$lon_c <- centroid_coords$lon_deg
    centroids$lat_c <- centroid_coords$lat_deg
    
    # Remove original coordinates
    centroids <- subset(centroids, select = -c(x1, x2))
    
    return(centroids)
    
  }, error = function(e) {
    logerror("Error in find_centroid for grid %d: %s", i, e$message)
    return(data.frame())
  })
}


#' Save results to CSV file
#' 
#' @param output_df Data frame to save
#' @param dggs_res DGGS resolution
#' @param fid Grid ID
save_results <- function(output_df, dggs_res, fid) {
  # Create output directory
  output_dir <- sprintf("Result/Level%d/Centroid", dggs_res)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save to CSV
  output_file <- sprintf("%s/vege_pre_%d.csv", output_dir, fid)
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
    
    # Get resolution parameters
    params <- get_resolution_params(dggs_res)
    dggs_cellsize <- params$cell_size
    vertical_res <- params$vertical_res
    
    loginfo("Cell size: %f, Vertical resolution: %d", dggs_cellsize, vertical_res)
    
    # Create DGGS object
    dgg <- create_dggs(dggs_res)
    
    # Generate nested grids
    rectangles <- nested_grids(ncores, fid)
    
    # Record timing
    tic("Centroid generation")
    
    # Conduct parallel processing
    output_df <- foreach(
      i = c(1:ncores), 
      .combine = rbind,
      .packages = 'dggridR'
    ) %dopar% {
      find_centroid(i, dggs_cellsize, dgg, rectangles)
    }
    
    # Remove duplicates
    output_df <- output_df[!duplicated(output_df$Cell_address), ]
    
    # Record timing
    toc()
    
    # Save results
    save_results(output_df, dggs_res, fid)
    
    loginfo("Centroid generation completed successfully")
    
  }, error = function(e) {
    logerror("Script failed: %s", e$message)
    quit(status = 1)
  })
}


# Run main function if script is executed directly
if (!interactive()) {
  main()
}
