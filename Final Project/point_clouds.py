import numpy as np
import os
import open3d as o3d
import glob
import re
import argparse
from tqdm import tqdm
import time
import concurrent.futures
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import umap
import multiprocessing
import warnings
import gc
from multiprocessing import Manager, Pool
from functools import partial
import platform
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data_efficiently(file_path):
    """Load data from npz file with memory efficiency optimizations"""
    # Use np.load with mmap_mode for memory-mapped access which is more efficient for large files
    try:
        # First try to open with memory mapping for large files
        with np.load(file_path, mmap_mode='r') as data:
            # Load complete data, but still make a copy to release the memory map
            points = data['points'].copy()
            colors = data['colors'].copy()
    except Exception as e:
        # Fall back to standard loading if memory mapping fails
        data = np.load(file_path)
        points = data['points']
        colors = data['colors']
    
    # Handle NaN or Inf values
    valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Return only the valid data
    return points, colors

def extract_frame_info(filename):
    """Extract frame number and life stage from filename"""
    match = re.search(r'frame_(\d+)-(\d+)', filename)
    if match:
        frame = int(match.group(1))
        life_stage = int(match.group(2))
        return frame, life_stage
    return None, None

def process_file(file_path, out_folder, file_index, max_points=50000, use_incremental_pca=False):
    """Process a single file with dimensionality reduction and save as point cloud"""
    try:
        # Extract frame and life stage info for logging purposes
        frame, life_stage = extract_frame_info(os.path.basename(file_path))
        if frame is None:
            return f"Error: Could not parse frame info from {file_path}"
        
        # Efficient data loading with potential subsampling
        points, colors = load_data_efficiently(file_path)
        
        # Check for valid data
        if len(points) == 0:
            return f"Warning: No points found in {file_path}"
        
        # Apply dimensionality reduction with memory efficiency
        # First apply PCA on combined points and colors
        combined_data = np.hstack([points, colors])
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(combined_data)
        
        # Clear combined_data from memory
        del combined_data
        gc.collect()
        
        # Then apply UMAP on PCA results with optimized parameters
        reducer = umap.UMAP(
            init='pca',           # Initialize with PCA results
            n_neighbors=10,       # Local vs global relationship
            min_dist=0.1,         # Determines clustering
            n_components=3,       # 3D output
            metric='euclidean',   # Fastest distance metric
            low_memory=True,      # Optimize for memory usage
            n_jobs=-1,            # Use all available cores
            random_state=None     # Allow parallelism
        )
        reducer_coords = reducer.fit_transform(pca_coords)
        
        # Clear PCA results from memory
        del pca_coords
        gc.collect()
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reducer_coords)
        
        # Make sure colors are properly formatted for visualization
        color_array = colors[:len(reducer_coords)]
        
        # If colors are in range [0,255], normalize to [0,1]
        if np.max(color_array) > 1.0:
            color_array = color_array.astype(float) / 255.0
            
        # Ensure values are clipped to valid range [0,1]
        color_array = np.clip(color_array, 0.0, 1.0)
        
        # Set colors
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        
        # Save as ply file with sequential numbering
        out_path = f"{out_folder}/{file_index}.ply"
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=True, compressed=False)
        
        # Force cleanup
        del pcd, reducer_coords, colors, points
        gc.collect()
        
        return f"Successfully processed {os.path.basename(file_path)} as {file_index}.ply"
    
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def pipeline_process_batch(batch_files, out_folder, start_index, max_points=50000, use_incremental_pca=False):
    """Process a batch of files in a pipeline fashion"""
    results = []
    for i, file_path in enumerate(batch_files):
        file_index = start_index + i
        result = process_file(file_path, out_folder, file_index, max_points, use_incremental_pca)
        results.append(result)
    return results

def process_file_with_args(args):
    """Wrapper function to unpack arguments for process_file to work with multiprocessing"""
    return process_file(*args)

def worker_process_init():
    """Initialize worker process - Windows-compatible process initialization"""
    # Set process priority only on non-Windows platforms
    # Windows has a different process priority system that requires elevated privileges
    if platform.system() != 'Windows':
        try:
            import psutil
            p = psutil.Process()
            p.nice(10)  # Lower priority (higher nice value)
        except ImportError:
            pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process point cloud data with dimensionality reduction')
    parser.add_argument('--in-folder', type=str, default='../../MaNGA/frames2', help='Input folder containing npz files')
    parser.add_argument('--out-folder', type=str, default='../../MaNGA/pointclouds', help='Output folder for point cloud files')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count()-1), help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for parallel processing')
    parser.add_argument('--max-points', type=int, default=50000, help='Maximum number of points to process per file (0 for no limit)')
    parser.add_argument('--pipeline', action='store_true', help='Use pipeline parallelism (recommended)')
    parser.add_argument('--no-incremental', action='store_true', help='Disable incremental PCA')
    parser.add_argument('--sequential', action='store_true', help='Force sequential processing')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    
    # Get all npz files in the input folder
    files = glob.glob(f"{args.in_folder}/frame_*.npz")
    if not files:
        print(f"No npz files found in {args.in_folder}")
        return
    
    # Sort files by frame (descending) and life stage (ascending)
    file_info = []
    for file_path in files:
        frame, life_stage = extract_frame_info(os.path.basename(file_path))
        if frame is not None:
            file_info.append((file_path, frame, life_stage))
    
    # Sort by frame (descending) and life stage (ascending)
    file_info.sort(key=lambda x: (-x[1], x[2]))
    sorted_files = [info[0] for info in file_info]
    
    print(f"Found {len(sorted_files)} files to process")
    print(f"Processing in order: {os.path.basename(sorted_files[0])} to {os.path.basename(sorted_files[-1])}")
    
    # Force sequential if requested
    if args.sequential:
        args.parallel = False
        args.pipeline = False
    
    print(f"Using {'pipeline parallelism' if args.pipeline else 'standard parallelism' if args.parallel else 'sequential processing'}")
    
    # Process files
    results = []
    file_counter = 1  # Counter for sequential file naming
    
    if (args.parallel or args.pipeline) and not args.sequential:
        try:
            # For pipeline mode, we use a different parallel strategy
            if args.pipeline:
                # Create a pool (without initialization function on Windows)
                if platform.system() == 'Windows':
                    pool = Pool(processes=args.workers)
                else:
                    pool = Pool(processes=args.workers, initializer=worker_process_init)
                
                try:
                    manager = Manager()
                    
                    # Process in larger chunks for pipeline parallelism
                    pipeline_batch_size = args.batch_size * 2
                    
                    # Prepare batches
                    batches = []
                    for i in range(0, len(sorted_files), pipeline_batch_size):
                        batch = sorted_files[i:i+pipeline_batch_size]
                        start_idx = file_counter + i
                        batches.append((batch, args.out_folder, start_idx, args.max_points, not args.no_incremental))
                    
                    # Submit batches to the pool
                    for i, batch_result in enumerate(tqdm(
                        pool.starmap(pipeline_process_batch, batches),
                        total=len(batches),
                        desc="Processing batches"
                    )):
                        results.extend(batch_result)
                        
                        # Force garbage collection between batches
                        gc.collect()
                        
                    # Update counter
                    file_counter += len(sorted_files)
                finally:
                    pool.close()
                    pool.join()
            else:
                # Standard process pool approach with smaller batches
                for i in range(0, len(sorted_files), args.batch_size):
                    batch = sorted_files[i:i+args.batch_size]
                    batch_indices = list(range(file_counter, file_counter + len(batch)))
                    print(f"Processing batch {i//args.batch_size + 1}/{len(sorted_files)//args.batch_size + 1}")
                    
                    # Create process arguments
                    process_args = [
                        (
                            file_path,
                            args.out_folder,
                            file_index,
                            args.max_points,
                            not args.no_incremental
                        )
                        for file_path, file_index in zip(batch, batch_indices)
                    ]
                    
                    # On Windows, don't use the initializer
                    if platform.system() == 'Windows':
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=min(args.workers, len(batch))
                        ) as executor:
                            batch_results = list(tqdm(
                                executor.map(process_file_with_args, process_args),
                                total=len(batch)
                            ))
                            results.extend(batch_results)
                    else:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=min(args.workers, len(batch)),
                            initializer=worker_process_init
                        ) as executor:
                            batch_results = list(tqdm(
                                executor.map(process_file_with_args, process_args),
                                total=len(batch)
                            ))
                            results.extend(batch_results)
                    
                    # Update the counter for the next batch
                    file_counter += len(batch)
                    
                    # Force garbage collection between batches
                    gc.collect()
                
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing
            file_counter = 1  # Reset counter
            for file_path in tqdm(sorted_files, desc="Processing"):
                results.append(process_file(
                    file_path, 
                    args.out_folder,
                    file_counter,
                    args.max_points,
                    not args.no_incremental
                ))
                file_counter += 1
    else:
        # Sequential processing
        for file_path in tqdm(sorted_files, desc="Processing"):
            results.append(process_file(
                file_path, 
                args.out_folder,
                file_counter,
                args.max_points,
                not args.no_incremental
            ))
            file_counter += 1
            
            # Force garbage collection every 10 files
            if file_counter % 10 == 0:
                gc.collect()
    
    # Report results
    success = [r for r in results if r.startswith("Successfully")]
    warnings = [r for r in results if r.startswith("Warning")]
    errors = [r for r in results if r.startswith("Error")]
    
    print(f"\nProcessed {len(results)} files")
    print(f"Successful: {len(success)}")
    
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds ({elapsed_time/len(sorted_files):.2f} seconds per file)")

if __name__ == "__main__":
    # To run with default settings (sequential):
    #   python script_name.py
    # To run with standard parallelism:
    #   python script_name.py --parallel
    # To run with pipeline parallelism (recommended for best performance):
    #   python script_name.py --pipeline
    # To specify custom folders:
    #   python script_name.py --in-folder path/to/input --out-folder path/to/output
    # Force sequential mode:
    #   python script_name.py --sequential
    main()