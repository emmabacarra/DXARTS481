import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time
import argparse

def process_frame_stage(frame, life_stage, frame_subset, base_dir, output_dir=None):
    """Process a single frame and life stage combination"""
    identity = f'{frame}-{life_stage+1}'
    file_path = f'{base_dir}/frames2/frame_{identity}.npz'
    
    # Determine output path - either same as input or a new directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = f'{output_dir}/frame_{identity}.npz'
    else:
        out_path = file_path
    
    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    try:
        # Load the data 
        data = np.load(file_path)
        points = data['points']
        colors = data['colors']
        
        # Initialize arrays to store new data
        new_points = []
        new_colors = []
        
        # Current position in the array
        start_index = 0
        
        # Process each object in this frame
        for _, obj in frame_subset.iterrows():
            adjusted_length = int(obj['Adjusted_Length'])
            pixels = adjusted_length**2
            
            # Calculate end index for the base layer (first layer)
            end_index = start_index + pixels
            
            # Make sure we don't go out of bounds
            if end_index <= len(points):
                # Add only the first layer of points and colors
                new_points.append(points[start_index:end_index])
                new_colors.append(colors[start_index:end_index])
                
                # Skip the remaining depth layers (99 more layers)
                start_index = end_index + (pixels * 99)
            else:
                print(f"Warning: Index out of bounds for {identity}, needed {end_index} but array length is {len(points)}")
                # Skip to the next object
                break
        
        # Stack the points and colors
        if new_points:
            new_points = np.vstack(new_points)
            new_colors = np.vstack(new_colors)
            
            # Remove any infinities
            valid_mask = ~np.any(np.isinf(new_points), axis=1)
            new_points = new_points[valid_mask]
            new_colors = new_colors[valid_mask]
            
            # Sample down if we have too many points
            if len(new_points) > 50000:
                indices = np.random.choice(len(new_points), 50000, replace=False)
                new_points = new_points[indices]
                new_colors = new_colors[indices]
            
            # Pad if we don't have enough points
            if len(new_points) < 50000:
                padding = 50000 - len(new_points)
                new_points = np.pad(new_points, ((0, padding), (0, 0)), mode='constant', constant_values=0)
                new_colors = np.pad(new_colors, ((0, padding), (0, 0)), mode='constant', constant_values=0)
            
            # Verify shapes
            assert new_points.shape == (50000, 3), f"Points shape error: {new_points.shape}"
            assert new_colors.shape == (50000, 3), f"Colors shape error: {new_colors.shape}"
            
            # Save the processed data
            np.savez_compressed(out_path, points=new_points, colors=new_colors)
            return f"Successfully processed frame {identity}"
        else:
            return f"No valid points found for frame {identity}"
            
    except Exception as e:
        return f"Error processing {identity}: {str(e)}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process galaxy data frames')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of worker processes')
    parser.add_argument('--resume', type=int, default=679, help='Frame to resume from')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: overwrite input)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for parallel processing')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load data
    adjusted_info_data = pd.read_csv('adjusted_info_data.csv')
    n_frames_actual = len(np.unique(adjusted_info_data['Frame']))
    base_dir = '../../MaNGA'
    
    # Get frames to process
    frames = np.unique(adjusted_info_data['Frame']) if args.resume == 0 else range(args.resume, 680)
    
    # Create task list
    tasks = []
    for frame in frames:
        frame_subset = adjusted_info_data[adjusted_info_data['Frame'] == frame]
        for life_stage in range(6):
            tasks.append((frame, life_stage, frame_subset, base_dir, args.output))
    
    print(f"Processing {len(tasks)} frame-stage combinations...")
    
    # Process tasks
    results = []
    
    if args.parallel:
        try:
            import concurrent.futures
            # Process in smaller batches to avoid memory issues
            for i in range(0, len(tasks), args.batch_size):
                batch = tasks[i:i+args.batch_size]
                print(f"Processing batch {i//args.batch_size + 1}/{len(tasks)//args.batch_size + 1}")
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(batch))) as executor:
                    batch_results = list(tqdm(executor.map(
                        lambda args: process_frame_stage(*args), 
                        batch
                    ), total=len(batch)))
                    results.extend(batch_results)
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing
            for frame, life_stage, frame_subset, base_dir, output_dir in tqdm(tasks, desc="Processing"):
                results.append(process_frame_stage(frame, life_stage, frame_subset, base_dir, output_dir))
    else:
        # Sequential processing
        for frame, life_stage, frame_subset, base_dir, output_dir in tqdm(tasks, desc="Processing"):
            results.append(process_frame_stage(frame, life_stage, frame_subset, base_dir, output_dir))
    
    # Report results
    success = [r for r in results if r.startswith("Successfully")]
    errors = [r for r in results if not r.startswith("Successfully")]
    
    print(f"\nProcessed {len(results)} frames")
    print(f"Successful: {len(success)}")
    
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # To run with default settings (sequential processing):
    #   python script_name.py
    # To run with parallel processing:
    #   python script_name.py --parallel
    # To specify output directory:
    #   python script_name.py --output ../../MaNGA/frames2_fixed
    main()