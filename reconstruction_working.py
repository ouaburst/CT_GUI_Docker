# Script filename: reconstruction_working.py
from __future__ import annotations

# Standard library
import argparse
import json
import time
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import odl

# Local helper functions from the geometry module
from odl_utils_working import (
    build_working_geometry,
    load_json,
    print_debug,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    This function defines all user-configurable inputs for the script.
    It returns an argparse.Namespace object, so later in the script we can access:
        args.data_dir
        args.metadata_path
        args.output_dir
        args.reconstruction_method
        ...

    Main options:
    - --data_dir:
        Folder containing the input files such as:
            angles.npy
            axial_positions.npy
            sinogram.npy
    - --metadata_path:
        Path to metadata.json describing detector and reconstruction settings.
    - --output_dir:
        Folder where reconstructed results will be saved.
    - --reconstruction_method:
        Which reconstruction algorithm to run:
            adjoint
            fbp
            landweber
    - --parameters:
        Extra method-specific parameters as a JSON string.
        Example for FBP:
            '{"filter_type":"Ram-Lak","frequency_scaling":1.0,"padding":true}'
        Example for Landweber:
            '{"niter":20,"omega":0.5}'
    - --save_png:
        If present, save central slice PNG previews.
    - --progress_every:
        For iterative methods like Landweber, print progress every N iterations.
    """
    ap = argparse.ArgumentParser(description="Standalone reconstruction using original working geometry")

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--metadata_path", type=str, default="metadata.json")
    ap.add_argument("--output_dir", type=str, default="reconstruction_working")

    ap.add_argument(
        "--reconstruction_method",
        type=str,
        default="fbp",
        choices=["adjoint", "fbp", "landweber"],
    )
    ap.add_argument("--parameters", type=str, default="{}")
    ap.add_argument("--save_png", action="store_true")

    ap.add_argument("--progress_every", type=int, default=5)
    return ap.parse_args()


def write_nrrd(output_path: Path, volume: np.ndarray, voxel_sizes_mm, min_pt_xyz):
    """
    Save a reconstructed 3D volume to NRRD format.

    Parameters
    ----------
    output_path : Path
        Output filename, for example:
            reconstruction_working/pine_16_1_fbp_working.nrrd

    volume : np.ndarray
        3D reconstructed array with shape:
            (nx, ny, nz)

    voxel_sizes_mm : iterable
        Physical voxel size in mm along x, y, z.
        Usually obtained from:
            bundle.reco_space.cell_sides

    min_pt_xyz : iterable
        Physical origin of the reconstruction volume in world coordinates.
        Usually obtained from:
            bundle.reco_space.min_pt

    What this function does
    -----------------------
    - Builds a NRRD header containing:
        - volume size
        - voxel spacing
        - physical origin
    - Writes the data as float32

    Why this matters
    ----------------
    The header ensures that 3D Slicer or other imaging tools know:
    - how large each voxel is
    - where the volume starts in physical space
    """
    sx, sy, sz = voxel_sizes_mm
    ox, oy, oz = min_pt_xyz

    header = {
        "space": "left-posterior-superior",
        "sizes": volume.shape,
        "space directions": [
            (float(sx), 0.0, 0.0),
            (0.0, float(sy), 0.0),
            (0.0, 0.0, float(sz)),
        ],
        "kinds": ["domain", "domain", "domain"],
        "space origin": (float(ox), float(oy), float(oz)),
        "endian": "little",
        "encoding": "raw",
    }

    nrrd.write(str(output_path), volume.astype(np.float32), header)


def save_middle_slices_png(volume: np.ndarray, out_prefix: Path):
    """
    Save the middle slice along each axis as PNG.

    Parameters
    ----------
    volume : np.ndarray
        3D reconstructed volume with shape (nx, ny, nz).

    out_prefix : Path
        Prefix used to name the PNG files.
        Example:
            out_prefix = Path("reconstruction_working/pine_16_1_fbp_working")

        Then the function will save:
            pine_16_1_fbp_working_x_mid.png
            pine_16_1_fbp_working_y_mid.png
            pine_16_1_fbp_working_z_mid.png

    What this function does
    -----------------------
    - Finds the center index along x, y, z
    - Extracts one slice for each direction
    - Saves a grayscale PNG for quick inspection

    Why this is useful
    ------------------
    You can quickly inspect reconstruction quality without opening the full NRRD.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Middle indices in the 3D array
    ix = volume.shape[0] // 2
    iy = volume.shape[1] // 2
    iz = volume.shape[2] // 2

    # Extract central slices
    slices = {
        "x_mid": volume[ix, :, :],
        "y_mid": volume[:, iy, :],
        "z_mid": volume[:, :, iz],
    }

    # Save each slice as a separate PNG
    for name, arr in slices.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(arr.T, cmap="gray", origin="lower")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(str(out_prefix.parent / f"{out_prefix.name}_{name}.png"), dpi=150)
        plt.close()


def main() -> None:
    """
    Main execution function.

    This is the full reconstruction pipeline of the script.

    Step-by-step overview
    ---------------------
    1. Parse command-line arguments.
    2. Create output directory.
    3. Load metadata.json.
    4. Load angles, axial positions, and sinogram from disk.
    5. Parse optional reconstruction parameters from JSON.
    6. Print input statistics for sanity checking.
    7. Build the working ODL geometry using the original-style implementation.
    8. Convert sinogram numpy array into an ODL range element.
    9. Run the selected reconstruction method:
         - adjoint
         - fbp
         - landweber
    10. Convert reconstruction result back to numpy.
    11. Print output statistics.
    12. Save reconstructed volume as NRRD.
    13. Optionally save center-slice PNGs.
    14. Save a JSON report with metadata and statistics.

    Reconstruction methods
    ----------------------
    adjoint:
        Backprojection only. Fast, but not a final reconstruction.
        Useful mainly for debugging and operator sanity checks.

    fbp:
        Filtered backprojection. Fast analytical reconstruction.
        Usually the main baseline.

    landweber:
        Simple iterative reconstruction:
            x_{k+1} = x_k + omega * A^T (b - A x_k)
        where:
            A   = forward operator
            A^T = adjoint
            b   = measured sinogram
        Slower than FBP, but can sometimes reduce artifacts.
    """
    # Read user inputs from command line
    args = parse_args()

    # Resolve important paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata dictionary from metadata.json
    metadata = load_json(args.metadata_path)

    # Load input arrays from disk
    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    sinogram = np.load(data_dir / "sinogram.npy")

    # Parse optional JSON parameters
    # Example:
    #   --parameters '{"filter_type":"Ram-Lak","frequency_scaling":1.0,"padding":true}'
    try:
        params = json.loads(args.parameters)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in --parameters: {e}") from e

    # Print input summary so we can verify that files and values look sensible
    print_debug("=" * 72)
    print_debug("[INFO] reconstruction_working.py")
    print_debug(f"[INFO] data_dir                 : {data_dir}")
    print_debug(f"[INFO] metadata_path           : {args.metadata_path}")
    print_debug(f"[INFO] output_dir              : {output_dir}")
    print_debug(f"[INFO] method                  : {args.reconstruction_method}")
    print_debug(f"[INFO] sinogram shape          : {sinogram.shape}")
    print_debug(f"[INFO] sinogram dtype          : {sinogram.dtype}")
    print_debug(
        f"[INFO] sinogram min/max/mean   : "
        f"{float(np.min(sinogram)):.6f} / {float(np.max(sinogram)):.6f} / {float(np.mean(sinogram)):.6f}"
    )
    print_debug(f"[INFO] parameters              : {params}")
    print_debug("=" * 72)

    # Build the reconstruction geometry and operators
    #
    # bundle contains:
    # - bundle.ray_trafo         : forward projector A
    # - bundle.ray_trafo_adjoint : adjoint/backprojector A^T
    # - bundle.reco_space        : reconstruction grid definition
    # - bundle.geometry          : ODL geometry object
    # - bundle.debug_info        : dictionary with geometry diagnostics
    bundle = build_working_geometry(
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        debug=True,
    )

    # Convert the raw sinogram array into an ODL element living in the operator range.
    #
    # np.ascontiguousarray ensures that the data layout is contiguous in memory,
    # which is safer for CUDA-based backends.
    sino_elem = bundle.ray_trafo.range.element(np.ascontiguousarray(sinogram, dtype=np.float32))

    # Start runtime measurement
    t0 = time.time()

    # ------------------------------------------------------------
    # Method 1: Adjoint
    # ------------------------------------------------------------
    if args.reconstruction_method == "adjoint":
        print_debug("[RUN] Applying adjoint ...")

        # A^T b
        # This is the backprojection of the measured sinogram.
        reco = bundle.ray_trafo_adjoint(sino_elem)

    # ------------------------------------------------------------
    # Method 2: FBP
    # ------------------------------------------------------------
    elif args.reconstruction_method == "fbp":
        print_debug("[RUN] Building FBP operator ...")

        # Build filtered backprojection operator from the forward model
        fbp = odl.tomo.fbp_op(bundle.ray_trafo, **params)

        print_debug("[RUN] Applying FBP ...")

        # Apply filtered backprojection to the sinogram
        reco = fbp(sino_elem)

    # ------------------------------------------------------------
    # Method 3: Landweber
    # ------------------------------------------------------------
    elif args.reconstruction_method == "landweber":
        # Read Landweber parameters from JSON.
        # Defaults:
        #   niter = 50
        #   omega = 0.5
        niter = int(params.get("niter", 50))
        omega = float(params.get("omega", 0.5))

        # Forward operator A
        A = bundle.ray_trafo

        # Initial image x_0 = 0
        x = A.domain.zero()

        print_debug(f"[RUN] Landweber: niter={niter}, omega={omega}")

        # How often to print progress
        pe = max(1, args.progress_every)

        # Iterative update:
        #   x <- x + omega * A^T (b - A x)
        for it in range(niter):
            resid = sino_elem - A(x)      # residual b - A x
            x += omega * A.adjoint(resid) # gradient-like update

            if ((it + 1) % pe == 0) or (it + 1 == niter):
                print_debug(f"[RUN] iter {it+1}/{niter}")

        reco = x

    # ------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown method: {args.reconstruction_method}")

    # Total runtime
    dt = time.time() - t0

    # Convert reconstructed ODL element back to a regular numpy array
    reco_np = reco.asarray().astype(np.float32)

    # Print output statistics
    print_debug(f"[OK] Reconstruction finished in {dt:.2f} s")
    print_debug("[SANITY] Reconstruction stats")
    print_debug(f"[SANITY] reco.shape             : {reco_np.shape}")
    print_debug(f"[SANITY] reco.dtype             : {reco_np.dtype}")
    print_debug(
        f"[SANITY] reco min/max/mean      : "
        f"{float(np.min(reco_np)):.6f} / {float(np.max(reco_np)):.6f} / {float(np.mean(reco_np)):.6f}"
    )

    # Base filename stem for outputs
    stem = f"{data_dir.name}_{args.reconstruction_method}_working"
    out_path = output_dir / f"{stem}.nrrd"

    # Save the reconstructed volume
    write_nrrd(
        out_path,
        reco_np,
        voxel_sizes_mm=bundle.reco_space.cell_sides,
        min_pt_xyz=bundle.reco_space.min_pt,
    )
    print_debug(f"[OK] NRRD saved to {out_path}")

    # Optionally save middle-slice preview images
    if args.save_png:
        save_middle_slices_png(reco_np, output_dir / stem)
        print_debug(f"[OK] PNGs saved with prefix {output_dir / stem}")

    # Create a JSON report summarizing what was done
    report = {
        "status": "ok",
        "data_dir": str(data_dir),
        "metadata_path": str(args.metadata_path),
        "method": args.reconstruction_method,
        "parameters": params,
        "runtime_seconds": dt,
        "sinogram_shape": list(sinogram.shape),
        "reconstruction_shape": list(reco_np.shape),
        "reconstruction_stats": {
            "min": float(np.min(reco_np)),
            "max": float(np.max(reco_np)),
            "mean": float(np.mean(reco_np)),
            "std": float(np.std(reco_np)),
        },
        "geometry_debug": bundle.debug_info,
        "output_nrrd": str(out_path),
    }

    # Save report to disk
    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_debug(f"[OK] Report saved to {report_path}")


# Standard Python entry point:
# this ensures that main() runs only when the file is executed directly
# and not when it is imported as a module.
if __name__ == "__main__":
    main()