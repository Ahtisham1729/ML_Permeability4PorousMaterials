import argparse
import csv
import gc
import glob
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# lbmpy imports
from lbmpy.session import *
from lbmpy.methods import create_trt_with_magic_number
from pystencils import Target

# ============== CONFIGURATION ==============
USE_GPU = True
USE_SINGLE_PRECISION = False
GPU_BLOCK_SIZE = (128, 1, 1)

# Default Directories (Override with arguments)
INPUT_DIR = ""
OUTPUT_DIR = ""

# Geometry settings
DATASET_NAME = "scalar_value"
SOLID_VALUE = 1
BODY_FORCE = 1e-5
# ===========================================


def load_geometry_from_file(path, flip=False, dataset_name="scalar_value", solid_value=1):
    """Load geometry from HDF5 file without downsampling."""
    with h5py.File(path, 'r') as file:
        if dataset_name not in file:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        raw = file[dataset_name][:]

    # Convert to binary mask (1=solid, 0=fluid)
    array = (raw == solid_value).astype(np.uint8)

    if flip:
        array = np.flip(array, axis=-1)

    return array, array.shape


def run_single_direction(geometry, domain_size, force_direction, force_magnitude,
                         use_gpu=True, use_single_precision=False):
    """Core simulation logic for one direction using TRT method."""
    force = [0.0, 0.0, 0.0]
    force[force_direction] = force_magnitude
    force = tuple(force)

    # Kinematic viscosity
    nu = 1/6

    # Calculate omega_plus from viscosity
    tau_plus = 3 * nu + 0.5
    omega_plus = 1.0 / tau_plus
    magic_lambda = 3/16
    tau_minus = magic_lambda / (tau_plus - 0.5) + 0.5
    omega_minus = 1.0 / tau_minus

    # ========================================
    # Step 3: Create config
    # ========================================
    lbm_config = LBMConfig(
        stencil=LBStencil(Stencil.D3Q19),
        method=Method.TRT,
        relaxation_rates=[omega_plus, omega_minus],
        force=force,
        force_model=ForceModel.GUO,
    )

    target = Target.GPU if use_gpu else Target.CPU
    config = CreateKernelConfig(
        target=target,
        gpu_indexing_params={"block_size": GPU_BLOCK_SIZE} if use_gpu else {},
        default_number_float="float32" if use_single_precision else "float64",
    )

    simulation = LatticeBoltzmannStep(
        domain_size=domain_size,
        periodicity=(True, True, True),
        lbm_config=lbm_config,
        config=config
    )

    # Boundary Conditions
    solid_bc = NoSlip(name="solid")
    mask = geometry.astype(bool)

    def geometry_callback(x, y, z):
        x_idx = np.floor(x).astype(int) % domain_size[0]
        y_idx = np.floor(y).astype(int) % domain_size[1]
        z_idx = np.floor(z).astype(int) % domain_size[2]
        return mask[x_idx, y_idx, z_idx]

    simulation.boundary_handling.set_boundary(
        solid_bc,
        mask_callback=geometry_callback,
        ghost_layers=True,
        inner_ghost_layers=True,
    )

    # Main Loop
    u_old = None
    stime = time.time()
    expected_shape = (*domain_size, 3)
    fluid_mask = ~mask
    converged = False
    final_iter = 0

    try:
        abs_tol = 1e-10       # Absolute change threshold
        rel_tol = 1e-6        # Relative change threshold
        for i in range(30):
            simulation.run(5000)

            u_current = np.array(simulation.velocity[:, :, :])
            if u_current.ndim == 4 and u_current.shape[1] == 1:
                u_current = u_current.reshape(expected_shape)

            if np.any(np.isnan(u_current)) or np.any(np.isinf(u_current)):
                return None

            if u_old is not None:
                numerator = np.sqrt(
                    np.sum(np.square(u_current - u_old)[fluid_mask]))
                denominator = np.sqrt(
                    np.sum(np.square(u_old)[fluid_mask])) + 1e-16
                epsilon = numerator / denominator
            else:
                epsilon = 1.0
                numerator = 1.0

            final_iter = i

            # Convergence check: relative OR absolute OR plateau
            if epsilon < rel_tol or numerator < abs_tol:
                converged = True
                break

            # Early divergence check
            if i > 9 and epsilon > 1e-2:
                return None

            u_old = u_current.copy()

        total_time = time.time() - stime

        u_x_mean = np.mean(u_current[..., 0][fluid_mask])
        u_y_mean = np.mean(u_current[..., 1][fluid_mask])
        u_z_mean = np.mean(u_current[..., 2][fluid_mask])

        return {
            'u_x': u_x_mean, 'u_y': u_y_mean, 'u_z': u_z_mean,
            'converged': converged, 'iterations': 5000 * (final_iter + 1),
            'time': total_time, 'nu': nu
        }

    finally:
        # Crucial for batch processing: release GPU memory
        del simulation
        gc.collect()


def simulate_full_tensor(geometry, domain_size, sample_name, force_magnitude=1e-5,
                         use_gpu=True, use_single_precision=False):
    mask = geometry.astype(bool)
    porosity = np.mean(~mask)

    results_by_direction = {}

    # Run X, Y, Z directions
    for direction in range(3):
        result = run_single_direction(geometry, domain_size, direction, force_magnitude,
                                      use_gpu, use_single_precision)
        if result is None:
            return None
        results_by_direction[direction] = result

    nu = results_by_direction[0]['nu']
    K_star = np.zeros((3, 3))

    # Construct Tensor
    for j in range(3):
        res = results_by_direction[j]
        K_star[0, j] = res['u_x'] * nu / force_magnitude
        K_star[1, j] = res['u_y'] * nu / force_magnitude
        K_star[2, j] = res['u_z'] * nu / force_magnitude

    # Darcy Permeability
    K = K_star * porosity

    # Eigenanalysis
    eigenvalues_K_star = np.linalg.eigvalsh(K_star)
    eigenvalues_K = np.linalg.eigvalsh(K)
    symmetry_error = np.linalg.norm(
        K_star - K_star.T) / (np.linalg.norm(K_star) + 1e-16)

    return {
        'sample_name': sample_name,
        'porosity': porosity,
        'nu': nu,
        'g': force_magnitude,
        'all_converged': all(results_by_direction[d]['converged'] for d in range(3)),
        'total_iterations': sum(results_by_direction[d]['iterations'] for d in range(3)),
        'iterations_x': results_by_direction[0]['iterations'],
        'iterations_y': results_by_direction[1]['iterations'],
        'iterations_z': results_by_direction[2]['iterations'],
        'K_star_xx': K_star[0, 0], 'K_star_xy': K_star[0, 1], 'K_star_xz': K_star[0, 2],
        'K_star_yx': K_star[1, 0], 'K_star_yy': K_star[1, 1], 'K_star_yz': K_star[1, 2],
        'K_star_zx': K_star[2, 0], 'K_star_zy': K_star[2, 1], 'K_star_zz': K_star[2, 2],
        'K_xx': K[0, 0], 'K_xy': K[0, 1], 'K_xz': K[0, 2],
        'K_yx': K[1, 0], 'K_yy': K[1, 1], 'K_yz': K[1, 2],
        'K_zx': K[2, 0], 'K_zy': K[2, 1], 'K_zz': K[2, 2],
        'K_star_eig1': eigenvalues_K_star[0], 'K_star_eig2': eigenvalues_K_star[1], 'K_star_eig3': eigenvalues_K_star[2],
        'K_eig1': eigenvalues_K[0], 'K_eig2': eigenvalues_K[1], 'K_eig3': eigenvalues_K[2],
        'symmetry_error': symmetry_error,
        'error_type': None,
        'error_message': None,
    }


def create_error_result(sample_name, error_type, error_message):
    """Create an error result dictionary with all fields matching successful results."""
    return {
        'sample_name': sample_name,
        'porosity': None,
        'nu': None,
        'g': BODY_FORCE,
        'all_converged': False,
        'total_iterations': None,
        'iterations_x': None,
        'iterations_y': None,
        'iterations_z': None,
        'K_star_xx': None, 'K_star_xy': None, 'K_star_xz': None,
        'K_star_yx': None, 'K_star_yy': None, 'K_star_yz': None,
        'K_star_zx': None, 'K_star_zy': None, 'K_star_zz': None,
        'K_xx': None, 'K_xy': None, 'K_xz': None,
        'K_yx': None, 'K_yy': None, 'K_yz': None,
        'K_zx': None, 'K_zy': None, 'K_zz': None,
        'K_star_eig1': None, 'K_star_eig2': None, 'K_star_eig3': None,
        'K_eig1': None, 'K_eig2': None, 'K_eig3': None,
        'symmetry_error': None,
        'error_type': error_type,
        # Truncate long messages
        'error_message': str(error_message)[:500] if error_message else None,
    }


def save_results_to_csv(results_list, output_path):
    if not results_list:
        return
    fieldnames = list(results_list[0].keys())

    # 'w' mode overwrites, assuming each sample has its own file in parallel mode
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)


def run_single_sample(filepath, output_dir):
    """Process a single geometry file and save INDIVIDUAL result."""
    sample_name = Path(filepath).stem
    individual_csv = os.path.join(output_dir, f"result_{sample_name}.csv")

    # Minimal logging with flush=True for SLURM
    print(f"START: {sample_name}", flush=True)

    try:
        geometry, domain_size = load_geometry_from_file(
            filepath, flip=True,
            dataset_name=DATASET_NAME, solid_value=SOLID_VALUE
        )

        results = simulate_full_tensor(
            geometry, domain_size, sample_name,
            force_magnitude=BODY_FORCE,
            use_gpu=USE_GPU, use_single_precision=USE_SINGLE_PRECISION
        )

        if results is not None:
            # Save successful result
            save_results_to_csv([results], individual_csv)
            print(f"DONE: {sample_name}", flush=True)
            return True
        else:
            # Save error result for convergence failure
            error_result = create_error_result(
                sample_name, "NoConvergence", "Simulation did not converge")
            save_results_to_csv([error_result], individual_csv)
            print(f"FAIL: {sample_name} (No Convergence)", flush=True)
            return False

    except Exception as e:
        # Save error result for exception
        error_result = create_error_result(
            sample_name, type(e).__name__, str(e))
        save_results_to_csv([error_result], individual_csv)
        print(f"ERR: {sample_name} | {e}", flush=True)
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-index', type=int, required=True,
                        help='Index of the sample to process (required for parallel mode)')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Sorting ensures index 0 maps to the same file on every node
    geometry_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    n_files = len(geometry_files)

    if n_files == 0:
        print(f"No .h5 files found in {args.input_dir}")
        exit(1)

    # Process single sample by index (parallel mode)
    if 0 <= args.sample_index < n_files:
        filepath = geometry_files[args.sample_index]
        success = run_single_sample(filepath, args.output_dir)
        exit(0 if success else 1)
    else:
        print(f"Index {args.sample_index} out of bounds (0-{n_files-1})")
        exit(1)
