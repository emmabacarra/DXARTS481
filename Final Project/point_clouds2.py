import os
import numpy as np
import cupy as cp
import pandas as pd
import open3d as o3d
import matplotlib.cm as cm
from skimage.transform import resize
from astropy import units as u, constants as const
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from scipy.stats import entropy as shannon_entropy
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from astropy.io import fits
from tqdm import tqdm
import concurrent.futures
import time

# Downsampling
def pooling(image_data, target_size=(9, 9), method='wrap'):
    return resize(image_data, target_size, preserve_range=True, mode=method)

# Natural jitter/breathing effect
def jitter_positions(points, intensity=0.1):
    # More efficient to generate all noise at once
    noise = np.random.normal(scale=intensity, size=points.shape)
    return points + noise

# Pre-define life cycle stages to avoid recreating this dictionary repeatedly
LIFE_CYCLE_STAGES = {
    'early formation': [0, 1, 2, 3, 4, 5, 8],
    'star formation': [7, 9, 10, 14, 19, 23, 24, 27, 28, 31, 33],
    'stellar evolution': [11, 12, 13, 22, 25, 26],
    'agn activity': [15, 16, 17, 20, 21],
    'late stage evolution': [18, 19, 25, 26, 30, 32, 34],
    'passive evolution': [11, 12, 13, 18, 19, 20, 21, 25, 26]
}

# Cache for isolation forest to avoid recomputing
iso_forest_cache = {}

def get_data(file_path, colormap=cm.managua_r, depth_scale_factor=5, spread_factor=0.5, 
             depth_layers=1, resample=None, pixels_only=False):
    """Optimized get_data function with caching and memory improvements"""
    if resample is not None:
        resize = resample
    else:
        resize = None
        
    kms = u.km / u.s
    mask_flags = 1 | 4 | 256 | 512
    
    # Using context manager properly
    with fits.open(file_path, memmap=True) as hdul:
        # Loading data more efficiently
        primary_header = hdul['PRIMARY'].header
        scinpvel = primary_header.get("SCINPVEL", 0) * kms  # Handle None case
        objra = primary_header.get("OBJRA", 0) * u.deg      # Handle None case
        objdec = primary_header.get("OBJDEC", 0) * u.deg    # Handle None case
        
        # Flux brightness for "color"
        g_flux_data = hdul['SPX_MFLUX'].data.copy()  # Use .copy() to avoid mmapped array issues
        flux_data = hdul['EMLINE_GFLUX'].data.copy()  # shape: (35, 74, 74)
        shape = flux_data.shape
        flux_mask = hdul['EMLINE_GFLUX_MASK'].data
        
        # Vectorize operations for better performance
        flux_masked = np.where(flux_data == 0 | ~np.isfinite(flux_data), 0, flux_data)
        flux_masked = np.where(np.bitwise_and(flux_mask, mask_flags) == 0, 0, flux_masked) # 1 means pixel unaffected
        flux_data = flux_masked.reshape(35, -1).T  # reshape to (35, 74*74) and transpose to (5476, 35)
        
        # Doppler shift for "depth"
        dshift_data = hdul['EMLINE_GVEL'].data[0] + scinpvel.value
        dshift_mask = hdul['EMLINE_GVEL_MASK'].data[0]
        dshift_data_masked = np.where((np.bitwise_and(dshift_mask, mask_flags) == 0) | ~np.isfinite(dshift_data), 0, dshift_data)
        dshift_max = np.nanmax(np.abs(dshift_data_masked.flatten()))
        if not np.isfinite(dshift_max) or dshift_max == 0:
            dshift_max = 1.0  # Use safe default
        depth = np.divide(dshift_data, dshift_max, out=np.zeros_like(dshift_data, dtype=float), where=dshift_max!=0) * depth_scale_factor
        
        # Spread of velocities from mean for "thickness"
        spread_map = hdul['EMLINE_GSIGMA'].data[23] * kms
        spread_min = np.nanmin(spread_map)
        spread_max = np.nanmax(spread_map) 
        # Avoid division by zero
        spread_norm = (spread_map - spread_min) / ((spread_max - spread_min))
        spread = spread_norm * spread_factor
        
        # Pixel coordinates
        skycoo_x = hdul['SPX_SKYCOO'].data[0] * u.arcsec
        skycoo_y = hdul['SPX_SKYCOO'].data[1] * u.arcsec
        delta_ra = (skycoo_x / np.cos(objdec)).to(u.deg)
        delta_dec = skycoo_y.to(u.deg)
        spaxel_ra = (objra + delta_ra)
        spaxel_dec = (objdec + delta_dec)
    
    # Object info
    redshift = scinpvel / const.c.to('km/s')  # redshift z = stellar velocity / speed of light
    distance = cosmo.luminosity_distance(redshift).to(u.Mpc)  # luminosity distance
    
    galaxy_coords = SkyCoord(ra=objra, dec=objdec, distance=np.abs(distance), frame='icrs')
    galaxy_x = galaxy_coords.cartesian.x.to(u.Mpc).value
    galaxy_y = galaxy_coords.cartesian.y.to(u.Mpc).value
    galaxy_z = galaxy_coords.cartesian.z.to(u.Mpc).value
    
    object_info = None
    if not pixels_only:
        plateifu = primary_header.get("PLATEIFU", None)
        age = cosmo.lookback_time(redshift).to('Gyr')  # lookback time of universe
        object_info = [plateifu, age.value, distance.value, redshift, scinpvel.value, [galaxy_x, galaxy_y, galaxy_z]]
    
    # Pixel info
    spaxel_coords = SkyCoord(ra=spaxel_ra, dec=spaxel_dec, frame='icrs')
    # Create a separate SkyCoord for the differentials to avoid copying large arrays
    diff_coords = SkyCoord(ra=spaxel_ra, dec=spaxel_dec, distance=np.abs(distance), frame='icrs')
    spaxel_cartesian = spaxel_coords.cartesian.with_differentials(diff_coords.cartesian.differentials)
    
    # Use pre-calculated indices instead of np.indices for better performance
    y_idx, x_idx = np.mgrid[:dshift_data.shape[0], :dshift_data.shape[1]]
    x = galaxy_x + x_idx + spaxel_cartesian.x.value
    y = galaxy_y + y_idx + spaxel_cartesian.y.value
    z = galaxy_z + depth + spaxel_cartesian.z.value
    
    # Use cached isolation forest if available
    flux_shape_key = flux_data.shape[0]
    if flux_shape_key not in iso_forest_cache:
        iso_forest = IsolationForest(contamination=0.30, max_samples=35, random_state=42, n_estimators=200)
        iso_forest_cache[flux_shape_key] = iso_forest
    else:
        iso_forest = iso_forest_cache[flux_shape_key]
    
    # Calculate features for isolation forest
    channels = [(shannon_entropy(flux_data[:, i]), np.std(flux_data[:, i])) for i in range(35)]
    labels = iso_forest.fit_predict(channels)
    selected_channels = np.where(labels == 1)[0]
    
    colors = []  # Final shape before tiling: (6, 5476, 3)
    for j, (stage, channels) in enumerate(LIFE_CYCLE_STAGES.items()):
        # Filter channels that are in both selected_channels and this stage
        selection = [c for c in channels if c in selected_channels]
        if not selection:  # Handle empty selection
            selection = channels[:1]  # Take first channel as fallback
            
        stage_flux = np.sum(flux_data[:, selection], axis=1)
        stage_flux = np.log1p(abs(stage_flux)).reshape(shape[1], shape[2])
        
        if resize is not None:
            downsampled_flux = pooling(stage_flux, target_size=(resize, resize))
            downsampled_g = pooling(g_flux_data, target_size=(resize, resize))
            stage_flux = downsampled_flux.flatten() + downsampled_g.flatten()
        else:
            stage_flux = stage_flux.flatten() + g_flux_data.flatten()
        
        # Normalize flux values to color range
        min_flux = np.min(stage_flux)
        max_flux = np.max(stage_flux)
        if not np.isfinite(min_flux) or not np.isfinite(max_flux) or max_flux == min_flux:
            min_flux = 0
            max_flux = 1
        
        norm_flux = np.divide(
            stage_flux - min_flux, 
            max_flux - min_flux,
            out=np.zeros_like(stage_flux, dtype=float),
            where=(max_flux - min_flux) != 0
        )
        norm_flux = np.nan_to_num(norm_flux, nan=0.5, posinf=1.0, neginf=0.0)
        stage_colors = colormap(norm_flux)[..., :3]  # mapping flux values to R, G, B (no alpha)
        colors.append(stage_colors)
        
    # Pre-allocate and fill arrays for better performance
    all_colors = np.tile(np.array(colors), (depth_layers, 1))
    
    # Prepare points with depth layers
    points_list = []
    for j in range(depth_layers):
        z_offset = spread.value * (j - depth_layers // 2)
        new_z = z + z_offset  # adding more layers of depth
        
        if resize is not None:
            downsample_x = pooling(x, target_size=(resize, resize))
            downsample_y = pooling(y, target_size=(resize, resize))
            downsample_z = pooling(new_z, target_size=(resize, resize))
            point_samples = np.column_stack([
                downsample_x.flatten(), 
                downsample_y.flatten(), 
                downsample_z.flatten()
            ])
        else:
            point_samples = np.column_stack([
                x.flatten(), 
                y.flatten(), 
                new_z.flatten()
            ])
        points_list.append(point_samples)
    
    # Stack all points
    all_points = np.vstack(points_list)
    
    pixel_info = [all_points, all_colors]
    
    if pixels_only:
        return pixel_info
    else:
        return object_info, pixel_info

def process_object(object_data, directory):
    """Process a single object and return its points and colors"""
    file_path = object_data["Filename"]
    adjusted_length = int(object_data["Adjusted_Length"])
    try:
        pixel_info = get_data(
            f'{directory}/raw/{file_path}',
            resample=adjusted_length, 
            pixels_only=True
        )
        points = cp.array(pixel_info[0], dtype=cp.float32)
        colors = cp.array(pixel_info[1], dtype=cp.float32)
        return points, colors
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def save_point_cloud(data, save_path):
    """Save a point cloud to disk"""
    frame, life_stage, points, colors = data
    
    identity = f'{frame}-{life_stage+1}'
    
    # Apply jitter to points
    life_points = jitter_positions(cp.asnumpy(points), intensity=0.4)
    life_colors = cp.asnumpy(colors)
    
    # Pad if necessary
    if len(life_points) < 50000:
        padding = 50000 - len(life_points)
        life_points = np.pad(life_points, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    
    if len(life_colors) < 50000:
        padding = 50000 - len(life_colors)
        life_colors = np.pad(life_colors, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    
    # Data reduction/analysis
    # Use fewer components for PCA to speed up initial dimensionality reduction
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(np.hstack([np.nan_to_num(life_points, nan=0), np.nan_to_num(life_colors, nan=0)]))
    
    # Use more efficient TSNE settings
    reducer = TSNE(
        n_components=3,
        perplexity=30,  # Adaptive perplexity
        n_iter=300,
        method='barnes_hut',  # Faster algorithm
        n_jobs=-1,
        random_state=42  # Fixed seed for reproducibility
    )
    reducer_coords = reducer.fit_transform(pca_coords)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reducer_coords)
    
    # Process colors
    color_array = life_colors[:len(reducer_coords)]
    if np.max(color_array) > 1.0:
        color_array = color_array.astype(float) / 255.0
    color_array = np.clip(color_array, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    # Save to disk
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True, compressed=False)
    return identity

def main():
    # Configuration
    directory = '../../MaNGA'
    save_folder = f'{directory}/pointclouds2'
    os.makedirs(save_folder, exist_ok=True)
    
    # Load data
    print("Loading adjusted info data...")
    adjusted_info_data = pd.read_csv('adjusted_info_data2.csv')
    n_frames_actual = len(np.unique(adjusted_info_data['Frame']))
    
    resume = 0
    k = 1
    
    # Process frames
    frame_values = np.unique(adjusted_info_data['Frame']) if resume == 0 else range(resume, n_frames_actual+1)
    
    print(f"Processing {len(frame_values)} frames...")
    for frame in tqdm(frame_values, desc="Frames"):
        frame_subset = adjusted_info_data[adjusted_info_data['Frame'] == frame]
        
        all_points = cp.empty((0, 3), dtype=cp.float32)
        all_colors = [cp.empty((0, 3), dtype=cp.float32) for _ in range(6)]
        
        # Process objects in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(frame_subset))) as executor:
            future_to_object = {
                executor.submit(process_object, object_data, directory): i 
                for i, (_, object_data) in enumerate(frame_subset.iterrows())
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_object), 
                              total=len(future_to_object), 
                              desc=f"Objects in frame {frame}/{n_frames_actual}",
                              leave=False):
                points, colors = future.result()
                if points is not None and colors is not None:
                    all_points = cp.vstack([all_points, points])
                    for life_stage in range(6):
                        all_colors[life_stage] = cp.vstack([all_colors[life_stage], colors[life_stage]])
        
        # Pad points if necessary
        if len(all_points) < 50000:
            padding = 50000 - len(all_points)
            all_points = cp.pad(all_points, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        
        # Process life stages in parallel
        point_cloud_tasks = []
        for life_stage in range(6):
            colors = all_colors[life_stage]
            out_path = f"{save_folder}/{k}.ply"
            point_cloud_tasks.append((frame, life_stage, all_points, colors))
            k += 1
        
        # Save point clouds in parallel to speed up I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            for i, task in enumerate(point_cloud_tasks):
                frame, life_stage, points, colors = task
                out_path = f"{save_folder}/{k-6+i}.ply"
                futures[executor.submit(save_point_cloud, task, out_path)] = (frame, life_stage+1)
            
            for future in concurrent.futures.as_completed(futures):
                identity = future.result()
                frame, life_stage = futures[future]
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")