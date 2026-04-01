# Script filename: reconstruction_working.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import odl

from odl_utils_working import (
    build_working_geometry,
    load_json,
    print_debug,
)


def parse_args() -> argparse.Namespace:
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
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    ix = volume.shape[0] // 2
    iy = volume.shape[1] // 2
    iz = volume.shape[2] // 2

    slices = {
        "x_mid": volume[ix, :, :],
        "y_mid": volume[:, iy, :],
        "z_mid": volume[:, :, iz],
    }

    for name, arr in slices.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(arr.T, cmap="gray", origin="lower")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(str(out_prefix.parent / f"{out_prefix.name}_{name}.png"), dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(args.metadata_path)

    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    sinogram = np.load(data_dir / "sinogram.npy")

    try:
        params = json.loads(args.parameters)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in --parameters: {e}") from e

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

    bundle = build_working_geometry(
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        debug=True,
    )

    sino_elem = bundle.ray_trafo.range.element(np.ascontiguousarray(sinogram, dtype=np.float32))

    t0 = time.time()

    if args.reconstruction_method == "adjoint":
        print_debug("[RUN] Applying adjoint ...")
        reco = bundle.ray_trafo_adjoint(sino_elem)

    elif args.reconstruction_method == "fbp":
        print_debug("[RUN] Building FBP operator ...")
        fbp = odl.tomo.fbp_op(bundle.ray_trafo, **params)
        print_debug("[RUN] Applying FBP ...")
        reco = fbp(sino_elem)

    elif args.reconstruction_method == "landweber":
        niter = int(params.get("niter", 50))
        omega = float(params.get("omega", 0.5))

        A = bundle.ray_trafo
        x = A.domain.zero()

        print_debug(f"[RUN] Landweber: niter={niter}, omega={omega}")
        pe = max(1, args.progress_every)

        for it in range(niter):
            resid = sino_elem - A(x)
            x += omega * A.adjoint(resid)
            if ((it + 1) % pe == 0) or (it + 1 == niter):
                print_debug(f"[RUN] iter {it+1}/{niter}")

        reco = x

    else:
        raise ValueError(f"Unknown method: {args.reconstruction_method}")

    dt = time.time() - t0
    reco_np = reco.asarray().astype(np.float32)

    print_debug(f"[OK] Reconstruction finished in {dt:.2f} s")
    print_debug("[SANITY] Reconstruction stats")
    print_debug(f"[SANITY] reco.shape             : {reco_np.shape}")
    print_debug(f"[SANITY] reco.dtype             : {reco_np.dtype}")
    print_debug(
        f"[SANITY] reco min/max/mean      : "
        f"{float(np.min(reco_np)):.6f} / {float(np.max(reco_np)):.6f} / {float(np.mean(reco_np)):.6f}"
    )

    stem = f"{data_dir.name}_{args.reconstruction_method}_working"
    out_path = output_dir / f"{stem}.nrrd"

    write_nrrd(
        out_path,
        reco_np,
        voxel_sizes_mm=bundle.reco_space.cell_sides,
        min_pt_xyz=bundle.reco_space.min_pt,
    )
    print_debug(f"[OK] NRRD saved to {out_path}")

    if args.save_png:
        save_middle_slices_png(reco_np, output_dir / stem)
        print_debug(f"[OK] PNGs saved with prefix {output_dir / stem}")

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

    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_debug(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    main()