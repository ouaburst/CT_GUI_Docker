"""
Purpose:
This script reconstructs a CT volume of a specified tree disk from the MITO dataset.
It supports adjoint, filtered back-projection (FBP), and Landweber iteration methods,
and saves the output as an NRRD file with appropriate metadata.

New:
- Verbose progress logging (incl. % for Landweber).
- Optional JSONL progress file via --progress_path (one line per update).
"""

import matplotlib.pyplot as plt
import nrrd
import os
import argparse
import odl
import json
import time
from dataset import MITO

# Set default matplotlib colormap to grayscale
plt.gray()

#########################################################
# main
# Parse CLI args, locate the requested sample in the MITO
# dataset, run the selected reconstruction method, and save
# the result as an NRRD with proper voxel spacing metadata.
#########################################################
def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser()
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

    # Load the dataset (defaults: mode='training', load_sinogram=True)
    dataset = MITO()

    # Search for the specified tree and disk
    for idx in range(len(dataset)):
        row = dataset.dataframe.iloc[idx]
        if row['tree_ID'] == args.tree_ID and row['disk_ID'] == args.disk_ID:
            sample = dataset[idx]  # dict with 'A', 'A_T', and 'sinogram' (if enabled)

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
                fbp_op = odl.tomo.fbp_op(sample['A'], **params)
                _progress("[FBP] Filtering/backprojecting …")
                reconstruction = fbp_op(sample['sinogram'])
                _progress("[FBP] Done.")

            elif args.reconstruction_method == 'landweber':
                # Retrieve iteration controls (with defaults)
                niter = int(params.get('niter', 50))
                omega = float(params.get('omega', 0.5))

                A = sample['A']
                # Ensure sinogram is an ODL element in A.range
                sinogram = A.range.element(sample['sinogram'])
                # Initialize reconstruction with zeros in the reconstruction space
                x = A.domain.zero()

                _progress(f"[LW] Starting Landweber: niter={niter}, omega={omega}")
                _progress("Checking space match…")
                _progress(f"  A.domain       : {A.domain}")
                _progress(f"  x.space        : {x.space}")
                _progress(f"  A.range        : {A.range}")
                _progress(f"  sinogram.space : {sinogram.space}")

                # Landweber iterative scheme:
                #   x_{k+1} = x_k + omega * A^*(b - A x_k)
                pe = max(1, args.progress_every)
                for it in range(niter):
                    resid = sinogram - A(x)   # residual in data space
                    x += omega * A.adjoint(resid)  # gradient step via adjoint

                    if ((it + 1) % pe == 0) or (it + 1 == niter):
                        pct = 100.0 * (it + 1) / niter
                        _progress(f"[LW] iter {it+1}/{niter}  ({pct:.1f}%)")

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
            break
    else:
        # Loop finished without 'break': sample not found
        raise FileNotFoundError(f"No sample matched tree_ID={args.tree_ID} and disk_ID={args.disk_ID}")

#########################################################
# Entrypoint
#########################################################
if __name__ == '__main__':
    main()