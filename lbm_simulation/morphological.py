#!/usr/bin/env python3
"""
Porous Media Morphological Feature Extraction

Calculates morphological features for porous microstructures:
- Specific surface (Sv)
- Tortuosity estimates

Designed for batch processing ~15,000 HDF5 geometry files.

"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.measure import marching_cubes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_geometry_from_h5(filepath: str, dataset_name: Optional[str] = None) -> np.ndarray:
    """
    Load binary geometry from HDF5 file.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file
    dataset_name : str, optional
        Name of the dataset within the HDF5 file.
        If None, uses the first dataset found or common names.
    
    Returns
    -------
    np.ndarray
        Binary array where 0 = pore, 1 = solid
    """
    with h5py.File(filepath, 'r') as f:
        if dataset_name is not None:
            data = f[dataset_name][:]
        else:
            # Try common dataset names first
            common_names = ['scalar_value', 'geometry', 'data', 'microstructure', 'structure', 'voxels', 'field']
            for name in common_names:
                if name in f.keys():
                    data = f[name][:]
                    break
            else:
                # Use first available dataset
                keys = list(f.keys())
                if len(keys) == 0:
                    raise ValueError(f"No datasets found in {filepath}")
                # Handle nested groups
                def find_first_dataset(group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            return item[:]
                        elif isinstance(item, h5py.Group):
                            result = find_first_dataset(item)
                            if result is not None:
                                return result
                    return None
                
                data = find_first_dataset(f)
                if data is None:
                    raise ValueError(f"No datasets found in {filepath}")
    
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Flip axis order from (z, y, x) to (x, y, z) when loading 3D data
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))

    # Handle different encodings
    unique_vals = np.unique(data)
    
    # Case 1: {-1, +1} encoding (spinodoid format: +1=solid, -1=pore)
    if len(unique_vals) == 2 and -1 in unique_vals and 1 in unique_vals:
        return (data == 1).astype(np.uint8)
    
    # Case 2: {0, 1} encoding (standard: 1=solid, 0=pore)
    elif len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
        return data.astype(np.uint8)
    
    # Case 3: Continuous values (level-set) - threshold at 0
    else:
        logger.warning(f"Continuous data detected in {filepath}, thresholding at 0 (>0 = solid)")
        return (data > 0).astype(np.uint8)


def calculate_specific_surface(pore_space: np.ndarray, voxel_size: float = 1.0) -> float:
    """
    Calculate specific surface area (Sv) - surface area per unit volume.
    
    Uses marching cubes for accurate surface area estimation.
    
    Parameters
    ----------
    pore_space : np.ndarray
        Binary array where True/1 = pore
    voxel_size : float
        Physical size of each voxel
    
    Returns
    -------
    float
        Specific surface area (1/length units)
    """
    try:
        # Marching cubes for surface area
        verts, faces, _, _ = marching_cubes(pore_space.astype(float), level=0.5, spacing=(voxel_size,)*3)
        
        # Calculate surface area from triangular faces
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # Cross product for triangle areas
        cross = np.cross(v1 - v0, v2 - v0)
        surface_area = 0.5 * np.sum(np.linalg.norm(cross, axis=1))
        
        # Total volume
        total_volume = np.prod(pore_space.shape) * (voxel_size ** 3)
        
        return surface_area / total_volume
    
    except Exception as e:
        logger.warning(f"Marching cubes failed: {e}, using voxel counting")
        # Fallback: count interface voxels
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(pore_space)
        interface = dilated & ~pore_space
        surface_area = np.sum(interface) * (voxel_size ** 2)
        total_volume = np.prod(pore_space.shape) * (voxel_size ** 3)
        return surface_area / total_volume



def calculate_tortuosity(pore_space: np.ndarray) -> dict:
    """
    Estimate tortuosity using multiple methods.
    
    1. Geometric tortuosity via medial axis path length
    2. Formation factor approximation (if connected)
    
    Parameters
    ----------
    pore_space : np.ndarray
        Binary array where True/1 = pore

    Returns
    -------
    dict
        Tortuosity estimates from different methods
    """
    from scipy.ndimage import label
    
    results = {
        'tortuosity_geometric_x': np.nan,
        'tortuosity_geometric_y': np.nan,
        'tortuosity_geometric_z': np.nan,
        'tortuosity_mean': np.nan,
        'is_percolating_x': False,
        'is_percolating_y': False,
        'is_percolating_z': False
    }
    
    try:
        # Check percolation and calculate geometric tortuosity for each axis
        tortuosities = []
        
        for ax, ax_name in enumerate(['x', 'y', 'z']):
            if ax >= pore_space.ndim:
                continue
                
            # Check if pore space percolates along this axis
            # Get inlet and outlet slices
            inlet_slice = [slice(None)] * pore_space.ndim
            outlet_slice = [slice(None)] * pore_space.ndim
            inlet_slice[ax] = slice(0, 1)
            outlet_slice[ax] = slice(-1, None)
            
            # Label connected components
            labeled = label(pore_space)[0]
            
            # Find labels at inlet and outlet
            inlet_labels = set(np.unique(labeled[tuple(inlet_slice)])) - {0}
            outlet_labels = set(np.unique(labeled[tuple(outlet_slice)])) - {0}
            
            # Check for percolation
            percolating_labels = inlet_labels & outlet_labels
            is_percolating = len(percolating_labels) > 0
            results[f'is_percolating_{ax_name}'] = is_percolating
            
            if is_percolating:
                # Estimate geometric tortuosity
                # Use chord length method as approximation
                L_straight = pore_space.shape[ax]
                
                # Calculate average path through pore space
                # Using medial axis approximation
                try:
                    # Average chord length
                    chord_lengths = []
                    for i in range(pore_space.shape[(ax+1) % 3]):
                        for j in range(pore_space.shape[(ax+2) % 3]):
                            idx = [slice(None)] * 3
                            idx[(ax+1) % 3] = i
                            idx[(ax+2) % 3] = j
                            line = pore_space[tuple(idx)]
                            
                            # Count continuous pore segments
                            if np.any(line):
                                # Simple path estimate
                                changes = np.diff(line.astype(int))
                                segments = np.sum(np.abs(changes)) / 2 + 1
                                if segments > 0:
                                    avg_segment = np.sum(line) / segments
                                    chord_lengths.append(avg_segment)
                    
                    if chord_lengths:
                        avg_chord = np.mean(chord_lengths)
                        # Tortuosity approximation
                        tau = L_straight / avg_chord if avg_chord > 0 else np.nan
                        tau = max(1.0, tau)  # Tortuosity >= 1
                        results[f'tortuosity_geometric_{ax_name}'] = tau
                        tortuosities.append(tau)
                
                except Exception as e:
                    logger.debug(f"Detailed tortuosity calc failed for axis {ax_name}: {e}")
                    # Fallback: simple estimate based on porosity
                    porosity = np.mean(pore_space)
                    tau_archie = porosity ** (-0.5)  # Archie's law approximation
                    results[f'tortuosity_geometric_{ax_name}'] = tau_archie
                    tortuosities.append(tau_archie)
        
        if tortuosities:
            results['tortuosity_mean'] = np.mean(tortuosities)
    
    except Exception as e:
        logger.warning(f"Tortuosity calculation failed: {e}")
    
    return results



def extract_all_features(filepath: str, dataset_name: Optional[str] = None, 
                         voxel_size: float = 1.0) -> dict:
    """
    Extract all morphological features from a single geometry file.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    dataset_name : str, optional
        Dataset name within HDF5
    voxel_size : float
        Physical voxel size
    
    Returns
    -------
    dict
        All extracted features
    """
    filename = os.path.basename(filepath)
    
    try:
        # Load geometry (0 = pore, 1 = solid)
        geometry = load_geometry_from_h5(filepath, dataset_name)
        
        # Convert to pore space (True = pore)
        pore_space = (geometry == 0)
        
        # Basic properties
        porosity = np.mean(pore_space)
        shape = geometry.shape
        
        features = {
            'filename': filename,
            'filepath': filepath,
            'shape_x': shape[0] if len(shape) > 0 else np.nan,
            'shape_y': shape[1] if len(shape) > 1 else np.nan,
            'shape_z': shape[2] if len(shape) > 2 else np.nan,
            'porosity': porosity,
            'solid_fraction': 1 - porosity
        }
        
        # Skip if no pores or no solid
        if porosity == 0 or porosity == 1:
            logger.warning(f"Trivial geometry in {filename}: porosity = {porosity}")
            features['error'] = 'trivial_geometry'
            return features
        
        # 1. Specific surface
        sv = calculate_specific_surface(pore_space, voxel_size)
        features['specific_surface_Sv'] = sv

        # 2. Tortuosity
        tort = calculate_tortuosity(pore_space)
        features.update(tort)
        
        features['error'] = None
        features['processing_status'] = 'success'
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        features = {
            'filename': filename,
            'filepath': filepath,
            'error': str(e),
            'processing_status': 'failed'
        }
    
    return features


def process_single_file(args):
    """Wrapper for parallel processing."""
    filepath, dataset_name, voxel_size = args
    return extract_all_features(filepath, dataset_name, voxel_size)


def main():
    parser = argparse.ArgumentParser(
        description='Extract morphological features from porous microstructure geometries',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Directory containing HDF5 files or glob pattern (e.g., "/data/*.h5")'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='morphological_features.csv',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default=None,
        help='Dataset name within HDF5 files (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '-v', '--voxel-size',
        type=float,
        default=1.0,
        help='Physical voxel size (affects Sv and pore size units)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=500,
        help='Save intermediate results every N files'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output file (skip already processed files)'
    )
    
    args = parser.parse_args()
    
    # Find all HDF5 files
    input_path = Path(args.input_path)
    
    if input_path.is_dir():
        h5_files = list(input_path.glob('**/*.h5')) + list(input_path.glob('**/*.hdf5'))
    elif '*' in str(input_path):
        h5_files = [Path(f) for f in glob.glob(str(input_path), recursive=True)]
    else:
        h5_files = [input_path]
    
    h5_files = [str(f) for f in h5_files]
    
    if not h5_files:
        logger.error(f"No HDF5 files found matching: {args.input_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(h5_files)} HDF5 files to process")
    
    # Check for resume
    processed_files = set()
    existing_results = []
    
    if args.resume and os.path.exists(args.output):
        logger.info(f"Resuming from existing file: {args.output}")
        existing_df = pd.read_csv(args.output)
        processed_files = set(existing_df['filepath'].tolist())
        existing_results = existing_df.to_dict('records')
        logger.info(f"Found {len(processed_files)} already processed files")
        
        # Filter out already processed
        h5_files = [f for f in h5_files if f not in processed_files]
        logger.info(f"Remaining files to process: {len(h5_files)}")
    
    if not h5_files:
        logger.info("All files already processed!")
        return
    
    # Prepare arguments for parallel processing
    process_args = [(f, args.dataset, args.voxel_size) for f in h5_files]
    
    # Determine number of workers
    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    
    logger.info(f"Processing with {n_workers} parallel workers")
    
    # Process files
    all_results = existing_results.copy()
    checkpoint_counter = 0
    
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, arg): arg[0] for arg in process_args}
        
        with tqdm(total=len(h5_files), desc="Extracting features") as pbar:
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Unhandled error for {filepath}: {e}")
                    all_results.append({
                        'filename': os.path.basename(filepath),
                        'filepath': filepath,
                        'error': str(e),
                        'processing_status': 'failed'
                    })
                
                pbar.update(1)
                checkpoint_counter += 1
                
                # Checkpoint save
                if checkpoint_counter >= args.checkpoint_interval:
                    logger.info(f"Checkpoint: saving {len(all_results)} results")
                    df = pd.DataFrame(all_results)
                    df.to_csv(args.output, index=False)
                    checkpoint_counter = 0
    
    # Final save
    df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    priority_cols = [
        'filename', 'filepath', 'processing_status', 'error',
        'shape_x', 'shape_y', 'shape_z', 'porosity', 'solid_fraction',
        'specific_surface_Sv',
        'tortuosity_mean', 'tortuosity_geometric_x', 'tortuosity_geometric_y', 'tortuosity_geometric_z',
        'is_percolating_x', 'is_percolating_y', 'is_percolating_z'
    ]
    
    # Get existing columns in priority order, then remaining
    ordered_cols = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]
    
    df.to_csv(args.output, index=False)
    
    elapsed = datetime.now() - start_time
    
    # Summary statistics
    success_count = len(df[df['processing_status'] == 'success'])
    fail_count = len(df[df['processing_status'] == 'failed'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {len(df)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Time elapsed: {elapsed}")
    logger.info(f"Average time per file: {elapsed / len(h5_files) if h5_files else 0}")
    logger.info(f"Results saved to: {args.output}")
    
    # Print feature summary for successful files
    if success_count > 0:
        success_df = df[df['processing_status'] == 'success']
        logger.info(f"\nFeature Summary:")
        logger.info(f"Porosity: {success_df['porosity'].mean():.4f} ± {success_df['porosity'].std():.4f}")
        logger.info(f"Specific Surface (Sv): {success_df['specific_surface_Sv'].mean():.4f} ± {success_df['specific_surface_Sv'].std():.4f}")
        logger.info(f"  Mean Tortuosity: {success_df['tortuosity_mean'].mean():.4f} ± {success_df['tortuosity_mean'].std():.4f}")


if __name__ == '__main__':
    main()
