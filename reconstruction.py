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

    # Small helper: print + optional JSONL write
    def _progress(msg: str):
        print(msg, flush=True)
        if args.progress_path:
            try:
                with open(args.progress_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "msg": msg}) + "\n")
            except Exception:
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

    # Load the dataset
    dataset = MITO()

    # Search for the specified tree and disk
    for idx in range(len(dataset)):
        row = dataset.dataframe.iloc[idx]
        if row['tree_ID'] == args.tree_ID and row['disk_ID'] == args.disk_ID:
            sample = dataset[idx]

            # === Start reconstruction timing ===
            start_time = time.time()

            # --- Reconstruction ---
            if args.reconstruction_method == 'adjoint':
                _progress("[ADJ] Applying A^T …")
                reconstruction = sample['A_T'](sample['sinogram'])
                _progress("[ADJ] Done.")

            elif args.reconstruction_method == 'fbp':
                _progress("[FBP] Building FBP operator …")
                fbp_op = odl.tomo.fbp_op(sample['A'], **params)
                _progress("[FBP] Filtering/backprojecting …")
                reconstruction = fbp_op(sample['sinogram'])
                _progress("[FBP] Done.")

            elif args.reconstruction_method == 'landweber':
                # Retrieve iteration controls
                niter = int(params.get('niter', 50))
                omega = float(params.get('omega', 0.5))

                A = sample['A']
                sinogram = A.range.element(sample['sinogram'])
                x = A.domain.zero()

                _progress(f"[LW] Starting Landweber: niter={niter}, omega={omega}")
                _progress("Checking space match…")
                _progress(f"  A.domain       : {A.domain}")
                _progress(f"  x.space        : {x.space}")
                _progress(f"  A.range        : {A.range}")
                _progress(f"  sinogram.space : {sinogram.space}")

                # Landweber: x_{k+1} = x_k + omega * A^*(b - A x_k)
                pe = max(1, args.progress_every)
                for it in range(niter):
                    resid = sinogram - A(x)
                    x += omega * A.adjoint(resid)

                    if ((it + 1) % pe == 0) or (it + 1 == niter):
                        pct = 100.0 * (it + 1) / niter
                        _progress(f"[LW] iter {it+1}/{niter}  ({pct:.1f}%)")

                reconstruction = x
                _progress("[LW] Done.")

            else:
                raise ValueError(f"Unknown reconstruction_method: {args.reconstruction_method}")

            # === End reconstruction timing ===
            duration = time.time() - start_time
            _progress(f"Reconstruction completed in {duration:.2f} seconds.")

            # Convert reconstruction to numpy array
            reconstruction_np = reconstruction.asarray()

            # Extract voxel sizes
            sx, sy, sz = reconstruction.space.cell_sides

            # Prepare NRRD header information
            header = {
                'space': 'left-posterior-superior',
                'sizes': reconstruction_np.shape,
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
        raise FileNotFoundError(f"No sample matched tree_ID={args.tree_ID} and disk_ID={args.disk_ID}")

if __name__ == '__main__':
    main()