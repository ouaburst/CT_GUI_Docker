# Script filename: reconstruction_v4.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import odl

from odl_utils_v4 import (
    load_json,
    build_global_helical_geometry_v4,
    print_debug,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global helical reconstruction prototype v4 with shifts.npy")

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--metadata_path", type=str, default="metadata_v2.json")
    ap.add_argument("--output_dir", type=str, default="reconstruction_v4")

    ap.add_argument("--reconstruction_method", type=str, default="adjoint", choices=["adjoint", "fbp"])
    ap.add_argument("--filter_type", type=str, default="Ram-Lak")
    ap.add_argument("--frequency_scaling", type=float, default=1.0)
    ap.add_argument("--padding", action="store_true")

    ap.add_argument("--rec_min_x", type=float, default=-80.0)
    ap.add_argument("--rec_max_x", type=float, default=130.0)
    ap.add_argument("--rec_min_y", type=float, default=-200.0)
    ap.add_argument("--rec_max_y", type=float, default=20.0)
    ap.add_argument("--rec_min_z", type=float, default=90.0)
    ap.add_argument("--rec_max_z", type=float, default=470.0)
    ap.add_argument("--rec_voxel_size", type=float, default=2.0)

    ap.add_argument("--use_curved_detector", action="store_true")
    ap.add_argument("--disable_src_shifts", action="store_true")
    ap.add_argument("--disable_det_shifts", action="store_true")

    ap.add_argument("--save_png", action="store_true")
    return ap.parse_args()


def write_nrrd(output_path: Path, volume: np.ndarray, voxel_sizes_mm, min_pt_xyz):
    sx, sy, sz = voxel_sizes_mm
    ox, oy, oz = min_pt_xyz
    header = {
        "space": "left-posterior-superior",
        "sizes": volume.shape,
        "space directions": [(float(sx), 0.0, 0.0), (0.0, float(sy), 0.0), (0.0, 0.0, float(sz))],
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


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(args.metadata_path)

    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    shifts = np.load(data_dir / "shifts.npy")
    sinogram = np.load(data_dir / "sinogram.npy")

    print_debug("=" * 70)
    print_debug("[INFO] reconstruction_v4.py")
    print_debug(f"[INFO] data_dir                 : {data_dir}")
    print_debug(f"[INFO] output_dir              : {output_dir}")
    print_debug(f"[INFO] method                  : {args.reconstruction_method}")
    print_debug(f"[INFO] sinogram shape          : {sinogram.shape}")
    print_debug(f"[INFO] shifts shape            : {shifts.shape}")
    print_debug(f"[INFO] reco voxel size [mm]    : {args.rec_voxel_size}")
    print_debug(f"[INFO] reco bounds X [mm]      : {args.rec_min_x} .. {args.rec_max_x}")
    print_debug(f"[INFO] reco bounds Y [mm]      : {args.rec_min_y} .. {args.rec_max_y}")
    print_debug(f"[INFO] reco bounds Z [mm]      : {args.rec_min_z} .. {args.rec_max_z}")
    print_debug(f"[INFO] curved detector         : {args.use_curved_detector}")
    print_debug(f"[INFO] src shifts enabled      : {not args.disable_src_shifts}")
    print_debug(f"[INFO] det shifts enabled      : {not args.disable_det_shifts}")
    print_debug("=" * 70)

    geom = build_global_helical_geometry_v4(
        sinogram_shape=sinogram.shape,
        angles=angles,
        axial_positions=axial_positions,
        shifts=shifts,
        metadata=metadata,
        rec_min_x=args.rec_min_x,
        rec_max_x=args.rec_max_x,
        rec_min_y=args.rec_min_y,
        rec_max_y=args.rec_max_y,
        rec_min_z=args.rec_min_z,
        rec_max_z=args.rec_max_z,
        rec_voxel_size=args.rec_voxel_size,
        use_curved_detector=args.use_curved_detector,
        use_src_shifts=not args.disable_src_shifts,
        use_det_shifts=not args.disable_det_shifts,
        debug=True,
    )

    sino_elem = geom.ray_trafo.range.element(np.ascontiguousarray(sinogram, dtype=np.float32))

    t0 = time.time()

    if args.reconstruction_method == "adjoint":
        print_debug("[RUN] Applying global adjoint ...")
        reco = geom.ray_trafo.adjoint(sino_elem)
    else:
        print_debug("[RUN] Building global FBP operator ...")
        fbp = odl.tomo.fbp_op(
            geom.ray_trafo,
            filter_type=args.filter_type,
            frequency_scaling=args.frequency_scaling,
            padding=args.padding,
        )
        print_debug("[RUN] Applying global FBP ...")
        reco = fbp(sino_elem)

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

    stem = (
        f"{data_dir.name}_{args.reconstruction_method}_global_v4"
        f"_vox{str(args.rec_voxel_size).replace('.', 'p')}"
        f"_z{str(args.rec_min_z).replace('.', 'p')}_{str(args.rec_max_z).replace('.', 'p')}"
        f"_curved{int(args.use_curved_detector)}"
        f"_srcshift{int(not args.disable_src_shifts)}"
        f"_detshift{int(not args.disable_det_shifts)}"
    )

    out_path = output_dir / f"{stem}.nrrd"
    write_nrrd(
        out_path,
        reco_np,
        voxel_sizes_mm=geom.reco_space.cell_sides,
        min_pt_xyz=geom.reco_space.min_pt,
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
        "runtime_seconds": dt,
        "reconstruction_shape": list(reco_np.shape),
        "reconstruction_stats": {
            "min": float(np.min(reco_np)),
            "max": float(np.max(reco_np)),
            "mean": float(np.mean(reco_np)),
            "std": float(np.std(reco_np)),
        },
        "geometry_debug": geom.debug_info,
        "output_nrrd": str(out_path),
    }

    with open(output_dir / f"{stem}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print_debug(f"[OK] Report saved to {output_dir / f'{stem}_report.json'}")


if __name__ == "__main__":
    main()