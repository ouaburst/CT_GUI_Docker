"""
Purpose:
This script reconstructs a CT volume of a specified tree disk from the MITO dataset.
It supports adjoint, filtered back-projection (FBP), and Landweber iteration methods,
and saves the output as an NRRD file with appropriate metadata.

New:
- Verbose progress logging (incl. % for Landweber).
- Optional JSONL progress file via --progress_path (one line per update).
"""

import nrrd
import os
import argparse
import odl
from odl.applications.tomo.geometry import Geometry
from odl.applications.tomo.geometry.conebeam import ConeBeamGeometry
from odl.applications.tomo.operators.ray_trafo import RayTransform
from odl import Operator
from odl.contrib.torch import OperatorModule
import json
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

#########################################################
# main
# Parse CLI args, locate the requested sample in the MITO
# dataset, run the selected reconstruction method, and save
# the result as an NRRD with proper voxel spacing metadata.
#########################################################
def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--specie', type=str, required=True, help="Tree species")
    parser.add_argument('--tree_ID', type=int, required=True, help="Tree identifier")
    parser.add_argument('--disk_ID', type=int, required=True, help="Disk identifier")
    parser.add_argument('--output_folder', type=str, default='images', help="Folder to save output images")
    parser.add_argument('--reconstruction_method', type=str, default='adjoint',
                        choices=['adjoint', 'fbp', 'landweber'], help="Reconstruction method to use")
    parser.add_argument('--parameters', type=str, default='{}',
                        help='Additional reconstruction parameters as JSON string')

    # New: progress controls
    parser.add_argument('--progress_every', type=int, default=5,
                        help='Print progress every N iterations for iterative methods')
    parser.add_argument('--progress_path', type=str, default='',
                        help='Optional path to write JSONL progress updates')


    args = parser.parse_args()

    #########################################################
    # _progress (local helper)
    # Print a message and optionally append a JSONL line with
    # a timestamp to --progress_path for external monitoring.
    #########################################################
    def _progress(msg: str):
        print(msg, flush=True)
        if args.progress_path:
            try:
                with open(args.progress_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "msg": msg}) + "\n")
            except Exception:
                # Best-effort logging only; ignore filesystem errors here
                pass

    # Print selected configuration
    _progress(f"Tree specie: {args.specie}")
    _progress(f"Tree ID: {args.tree_ID}")
    _progress(f"Disk ID: {args.disk_ID}")
    _progress(f"Output folder: {args.output_folder}")
    _progress(f"Reconstruction Method: {args.reconstruction_method}")
    _progress(f"Parameters: {args.parameters}")

    # Create output directory if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Parse parameters JSON
    try:
        params = json.loads(args.parameters)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in --parameters")

    sample = load_sample(args.specie, args.tree_ID, args.disk_ID, params)

    # === Start reconstruction timing ===
    start_time = time.time()

    # --- Reconstruction ---
    # Notes:
    # * 'A'   = forward projector (volume -> sinogram)
    # * 'A_T' = adjoint/backprojector (sinogram -> volume)
    # * For ODL ops, pass ODL space elements, not raw np arrays.
    if args.reconstruction_method == 'adjoint':
        _progress("[ADJ] Applying A^T …")
        # Simple backprojection: not an inverse, but a fast baseline
        reconstruction = sample['A_T'](sample['sinogram'])
        _progress("[ADJ] Done.")

    elif args.reconstruction_method == 'fbp':
        _progress("[FBP] Building FBP operator …")
        # ODL builds a filtered-backprojection operator compatible with A
        fbp_op = odl.applications.tomo.analytic.filtered_back_projection.fbp_op(sample['A'], 
                                                                                padding=params["padding"],
                                                                                filter_type=params["filter_type"], 
                                                                                frequency_scaling=params["frequency_scaling"])
        _progress("[FBP] Filtering/backprojecting …")
        reconstruction = fbp_op(sample['sinogram'])
        _progress("[FBP] Done.")
        
    elif args.reconstruction_method == 'landweber':
        # Retrieve iteration controls (with defaults)
        iterations = int(params.get('iterations', 50))
        omega = float(params.get('relaxation', 0.5))

        A = sample['A']
        # Ensure sinogram is an ODL element in A.range
        sinogram = A.range.element(sample['sinogram'])
        # Initialize reconstruction with zeros in the reconstruction space
        x = A.domain.zero()

        _progress(f"[LW] Starting Landweber: iterations={iterations}, omega={omega}")
        _progress("Checking space match…")
        _progress(f"  A.domain       : {A.domain}")
        _progress(f"  x.space        : {x.space}")
        _progress(f"  A.range        : {A.range}")
        _progress(f"  sinogram.space : {sinogram.space}")

        # Landweber iterative scheme:
        #   x_{k+1} = x_k + omega * A^*(b - A x_k)
        pe = max(1, args.progress_every)
        for it in range(iterations):
            resid = sinogram - A(x)   # residual in data space
            x += omega * A.adjoint(resid)  # gradient step via adjoint

            if ((it + 1) % pe == 0) or (it + 1 == iterations):
                pct = 100.0 * (it + 1) / iterations
                _progress(f"[LW] iter {it+1}/{iterations}  ({pct:.1f}%)")

        reconstruction = x
        _progress("[LW] Done.")

    else:
        # Should never fire due to argparse choices, but keep for safety
        raise ValueError(f"Unknown reconstruction_method: {args.reconstruction_method}")

    # === End reconstruction timing ===
    duration = time.time() - start_time
    _progress(f"Reconstruction completed in {duration:.2f} seconds.")

    # Convert reconstruction (ODL element) to numpy array
    reconstruction_np = reconstruction.asarray()

    # Extract voxel sizes from the reconstruction space
    sx, sy, sz = reconstruction.space.cell_sides

    # Prepare NRRD header information
    header = {
        'space': 'left-posterior-superior',
        'sizes': reconstruction_np.shape,                 # (nx, ny, nz)
        'space directions': [(sx, 0, 0), (0, sy, 0), (0, 0, sz)],
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'raw'
    }

    # Save the reconstructed volume as an NRRD file with method name
    output_filename = f"tree{args.tree_ID}_disk{args.disk_ID}_{args.reconstruction_method}.nrrd"
    output_path = os.path.join(args.output_folder, output_filename)

    nrrd.write(output_path, reconstruction_np, header)
    _progress(f"Saved NRRD volume to {output_path}.")

def load_sample(specie: str, tree_ID: int, disk_ID: int, params: dict) -> dict:
    # Construct sample path
    data_folder_path = Path('/media/Store-SSD/real_datasets/ml_ready')
    sample_path = data_folder_path.joinpath(f'{specie}_{tree_ID}_{disk_ID}')

    sinogram_min = params["SINOGRAM_MIN"]
    sinogram_max = params["SINOGRAM_MAX"]
    
    metadata = dict(json.load(open(sample_path.joinpath('metadata.json'))))

    # Load geometry information
    angles = np.load(sample_path.joinpath('angles.npy'), mmap_mode="r")[sinogram_min:sinogram_max]
    axial_positions = np.load(sample_path.joinpath('axial_positions.npy'), mmap_mode="r")[sinogram_min:sinogram_max]
    
    # Create operators
    A, A_T = make_operators(angles, axial_positions, metadata, params)

    # Package outputs
    data_dict = {
        'A'   : A,     # Forward projector (volume -> sinogram)
        'A_T' : A_T,   # Adjoint/back-projector (sinogram -> volume)
    }

    data_dict['sinogram'] = np.load(sample_path.joinpath('sinogram.npy'), mmap_mode="r")[sinogram_min:sinogram_max]

    return data_dict

def make_operators(
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: dict,
    reco_metadata: dict,
    torch = False
    ):
    """
    Make Operators
    Creates forward and adjoint projection operators using ODL
    based on angles, axial positions, and metadata.

    WHAT THIS RETURNS:
    forward_operator = A   : maps reconstruction space -> data space
    adjoint_operator = A_T : maps data space -> reconstruction space

    TYPICAL DOWNSTREAM USE:
    - Forward model in iterative schemes: A(x)
    - Gradient step through data fidelity: A_T(A(x) - y)
    - Physics layer inside DL models via OperatorModule

    PERFORMANCE NOTES:
    - If metadata selects an ASTRA CUDA backend in your geometry parser,
    these ops will execute on GPU. ODL manages the heavy lifting.
    - Memory layout matters: prefer float32 where possible, avoid 
    unnecessary copies between numpy/ODL/torch.
    """

    # Currently only ODL backend is supported
    assert metadata['GEOMETRY_ENGINE'] == 'ODL'

    geometry = parse_ODL_geometry(angles, axial_positions, metadata)
    reco_space = create_reconstruction_space(reco_metadata)
    forward_operator, adjoint_operator = create_ray_transforms(geometry, reco_space, reco_metadata.get("use_cache", False))

    if torch:
        forward_operator = OperatorModule(forward_operator)
        adjoint_operator = OperatorModule(adjoint_operator)
    
    # Sanity check
    assert forward_operator.domain == adjoint_operator.range
    assert forward_operator.range == adjoint_operator.domain

    return forward_operator, adjoint_operator

def parse_ODL_geometry(
        angles: np.ndarray,
        axial_positions: np.ndarray,
        metadata: dict,
        ):
    if metadata['GEOMETRY_NAME'] == 'ConeBeamGeometry':
        return parse_ODL_ConeBeamGeometry(angles, axial_positions, metadata)
    else:
        raise NotImplementedError

def create_ray_transforms(geometry: Geometry, reco_space: odl.DiscretizedSpace, use_cache: bool) -> tuple[Operator, Operator]:
        ray_transform = RayTransform(reco_space, geometry, impl='astra_cuda', use_cache = use_cache)
        ray_transform_adjoint = ray_transform.adjoint

        return ray_transform, ray_transform_adjoint

def create_reconstruction_space(metadata: dict) -> odl.DiscretizedSpace:
    # FIXME:
    REC_MIN_X, REC_MIN_Y, REC_MIN_Z = metadata['REC_MIN_X'], metadata['REC_MIN_Y'], metadata['REC_MIN_Z']
    REC_MAX_X, REC_MAX_Y, REC_MAX_Z = metadata['REC_MAX_X'], metadata['REC_MAX_Y'], metadata['REC_MAX_Z']
    REC_NPX_X, REC_NPX_Y, REC_NPX_Z = metadata['REC_NPX_X'], metadata['REC_NPX_Y'], metadata['REC_NPX_Z'] #int((REC_MAX_Z - REC_MIN_Z) // metadata['REC_PIC_SIZE'])
    
    reco_space = odl.uniform_discr(
        min_pt=[REC_MIN_X, REC_MIN_Y, REC_MIN_Z],
        max_pt=[REC_MAX_X, REC_MAX_Y, REC_MAX_Z],
        shape= [REC_NPX_X, REC_NPX_Y, REC_NPX_Z],
        dtype='float32')

    return reco_space

def parse_ODL_ConeBeamGeometry(
        angles : np.ndarray,
        axial_positions: np.ndarray,
        metadata: dict,
        ) -> ConeBeamGeometry:
    DET_X_MIN = metadata['DET_X_MIN']
    DET_X_MAX = metadata['DET_X_MAX']
    DET_NPX_X = metadata['DET_NPX_X']
    DET_Z_MIN = metadata['DET_Z_MIN']
    DET_Z_MAX = metadata['DET_Z_MAX']
    DET_NPX_Z = metadata['DET_NPX_Z']
    detector_partition = odl.uniform_partition(
            [DET_X_MIN, DET_Z_MIN],
            [DET_X_MAX, DET_Z_MAX],
            (DET_NPX_X, DET_NPX_Z))

    angles_increasing = np.unwrap(angles)
    angle_partition = odl.nonuniform_partition(angles_increasing)
    
    z_shift_func = interp1d(
        angles_increasing, axial_positions, kind="linear",
        bounds_error=False, fill_value=(axial_positions[0], axial_positions[-1]) # type: ignore The function has a special case for fill_value being a 2-tuple.
    )

    def shift_func(angle):
        # FIXME: Use
        #np.interp(angle, angles_increasing, z_corrected)
        res = np.zeros((len(angle), 3))
        res[:, 2] = z_shift_func(angle)
        return res

    geometry = ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=metadata["SRC_RADIUS"],
        det_radius=metadata["DET_RADIUS"],
        det_curvature_radius=(metadata["DET_CURVATURE_RADIUS"], None),
        pitch=0, # type: ignore The argument is a float, the function annotation is wrong.
        axis=[0, 0, 1],
        src_shift_func=shift_func,     # could be set to a function of angle if needed
        det_shift_func=shift_func,
        translation=[0, 0, 0],
    )

    return geometry

#########################################################
# Entrypoint
#########################################################
if __name__ == '__main__':
    main()