# Script filename: reconstruction_v2.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import odl

from odl_utils_v2 import (
    load_json,
    build_geometry_v2,
    print_debug,
    estimate_window_z_center_mm,
    estimate_window_angle_based_z_center_mm,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debuggable ODL reconstruction v2")

    ap.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Direct path to dataset folder, e.g. /media/Store-SSD/real_datasets/ml_ready/pine_16_1",
    )
    ap.add_argument("--tree_ID", type=int, default=None)
    ap.add_argument("--disk_ID", type=int, default=None)
    ap.add_argument(
        "--data_root",
        type=str,
        default="/media/Store-SSD/real_datasets/ml_ready",
        help="Used only if --data_dir is not given",
    )

    ap.add_argument("--metadata_path", type=str, default="metadata_v2.json")

    ap.add_argument("--output_dir", type=str, default="reconstruction")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Only build geometry and run sanity checks")

    ap.add_argument("--reconstruction_method", type=str, default="fbp", choices=["fbp", "adjoint"])
    ap.add_argument("--filter_type", type=str, default="Ram-Lak")
    ap.add_argument("--frequency_scaling", type=float, default=1.0)
    ap.add_argument("--padding", action="store_true")

    ap.add_argument("--proj_start", type=int, default=0)
    ap.add_argument("--proj_stop", type=int, default=800, help="Use a smaller number first for speed/debug")
    ap.add_argument("--det_x_start", type=int, default=None)
    ap.add_argument("--det_x_stop", type=int, default=None)
    ap.add_argument("--det_z_start", type=int, default=None)
    ap.add_argument("--det_z_stop", type=int, default=None)

    ap.add_argument("--z_center", type=float, default=0.0, help="Manual center of local reconstruction slab in mm")
    ap.add_argument("--z_half_width_mm", type=float, default=10.0, help="Half-width of local reconstruction slab in mm")
    ap.add_argument("--auto_z_center", action="store_true", help="Use midpoint axial position of selected window")
    ap.add_argument(
        "--auto_z_center_from_angle",
        action="store_true",
        help="Use angle-based z center estimate: z0_fit + (pitch/(2pi))*theta_mid",
    )
    ap.add_argument("--z0_fit_mm", type=float, default=-2.063, help="Offset used with --auto_z_center_from_angle")
    ap.add_argument("--pitch_mm_per_turn", type=float, default=40.0, help="Pitch value for diagnostic tests")

    return ap.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)

    if args.tree_ID is None or args.disk_ID is None:
        raise ValueError("Provide either --data_dir or both --tree_ID and --disk_ID")

    return Path(args.data_root) / f"pine_{args.tree_ID}_{args.disk_ID}"


def load_input_data(data_dir: Path):
    angles_path = data_dir / "angles.npy"
    axial_path = data_dir / "axial_positions.npy"
    sino_path = data_dir / "sinogram.npy"

    if not angles_path.exists():
        raise FileNotFoundError(f"Missing file: {angles_path}")
    if not axial_path.exists():
        raise FileNotFoundError(f"Missing file: {axial_path}")
    if not sino_path.exists():
        raise FileNotFoundError(f"Missing file: {sino_path}")

    angles = np.load(angles_path)
    axial_positions = np.load(axial_path)
    sinogram = np.load(sino_path)

    return angles, axial_positions, sinogram


def basic_sanity_checks(angles, axial_positions, sinogram, metadata):
    print_debug("[SANITY] Input arrays")
    print_debug(f"[SANITY] angles.shape            : {angles.shape}")
    print_debug(f"[SANITY] axial_positions.shape   : {axial_positions.shape}")
    print_debug(f"[SANITY] sinogram.shape          : {sinogram.shape}")
    print_debug(f"[SANITY] sinogram.dtype          : {sinogram.dtype}")
    print_debug(
        f"[SANITY] sinogram min/max/mean   : "
        f"{float(np.min(sinogram)):.6f} / {float(np.max(sinogram)):.6f} / {float(np.mean(sinogram)):.6f}"
    )

    if sinogram.ndim != 3:
        raise ValueError(f"Expected sinogram shape (num_proj, det_x, det_z), got {sinogram.shape}")

    if sinogram.shape[0] != len(angles):
        raise ValueError("Mismatch: sinogram.shape[0] != len(angles)")

    if sinogram.shape[0] != len(axial_positions):
        raise ValueError("Mismatch: sinogram.shape[0] != len(axial_positions)")

    if sinogram.shape[1] != metadata["DET_NPX_X"]:
        raise ValueError(f"Mismatch: sinogram.shape[1]={sinogram.shape[1]} != DET_NPX_X={metadata['DET_NPX_X']}")

    if sinogram.shape[2] != metadata["DET_NPX_Z"]:
        raise ValueError(f"Mismatch: sinogram.shape[2]={sinogram.shape[2]} != DET_NPX_Z={metadata['DET_NPX_Z']}")

    if not np.all(np.isfinite(sinogram)):
        raise ValueError("Sinogram contains non-finite values.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("Angles contain non-finite values.")
    if not np.all(np.isfinite(axial_positions)):
        raise ValueError("Axial positions contain non-finite values.")

    dtheta = np.diff(np.unwrap(angles.astype(np.float64)))
    print_debug(f"[SANITY] angle increment min/max : {float(dtheta.min()):.6e} / {float(dtheta.max()):.6e}")

    dz = np.diff(axial_positions.astype(np.float64))
    print_debug(f"[SANITY] axial increment min/max : {float(dz.min()):.6e} / {float(dz.max()):.6e}")


def crop_sinogram(
    sinogram: np.ndarray,
    proj_start: int,
    proj_stop: int,
    det_x_start: int | None,
    det_x_stop: int | None,
    det_z_start: int | None,
    det_z_stop: int | None,
) -> np.ndarray:
    x0 = 0 if det_x_start is None else det_x_start
    x1 = sinogram.shape[1] if det_x_stop is None else det_x_stop
    z0 = 0 if det_z_start is None else det_z_start
    z1 = sinogram.shape[2] if det_z_stop is None else det_z_stop

    sino_crop = sinogram[proj_start:proj_stop, x0:x1, z0:z1]
    return np.ascontiguousarray(sino_crop, dtype=np.float32)


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


def write_nrrd(output_path: Path, volume: np.ndarray, voxel_sizes_mm):
    sx, sy, sz = voxel_sizes_mm
    header = {
        "space": "left-posterior-superior",
        "sizes": volume.shape,
        "space directions": [(float(sx), 0.0, 0.0), (0.0, float(sy), 0.0), (0.0, 0.0, float(sz))],
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "raw",
    }
    nrrd.write(str(output_path), volume.astype(np.float32), header)


def _float_tag(x: float) -> str:
    return f"{x:.4f}".replace("-", "m").replace(".", "p")


def build_output_stem(
    dataset_name: str,
    method: str,
    proj_start: int,
    proj_stop: int,
    pitch_mm_per_turn: float,
    z_center_mode: str,
    z_half_width_mm: float,
) -> str:
    return (
        f"{dataset_name}_{method}"
        f"_proj{proj_start}_{proj_stop}"
        f"_pitch{_float_tag(pitch_mm_per_turn)}"
        f"_zmode-{z_center_mode}"
        f"_zhalf{_float_tag(z_half_width_mm)}"
    )


def main():
    args = parse_args()

    data_dir = resolve_data_dir(args)
    metadata = load_json(args.metadata_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    angles, axial_positions, sinogram = load_input_data(data_dir)

    auto_modes = int(bool(args.auto_z_center)) + int(bool(args.auto_z_center_from_angle))
    if auto_modes > 1:
        raise ValueError("Use only one of --auto_z_center or --auto_z_center_from_angle")

    if args.auto_z_center:
        effective_z_center = estimate_window_z_center_mm(
            axial_positions_full=axial_positions,
            proj_start=args.proj_start,
            proj_stop=args.proj_stop,
        )
        z_center_mode = "axial_midpoint"

    elif args.auto_z_center_from_angle:
        effective_z_center = estimate_window_angle_based_z_center_mm(
            angles_full=angles,
            proj_start=args.proj_start,
            proj_stop=args.proj_stop,
            pitch_mm_per_turn=args.pitch_mm_per_turn,
            z0_fit_mm=args.z0_fit_mm,
        )
        z_center_mode = "angle_based"

    else:
        effective_z_center = float(args.z_center)
        z_center_mode = "manual"

    print_debug("=" * 70)
    print_debug("[INFO] reconstruction_v2.py")
    print_debug(f"[INFO] data_dir                 : {data_dir}")
    print_debug(f"[INFO] metadata_path           : {args.metadata_path}")
    print_debug(f"[INFO] output_dir              : {output_dir}")
    print_debug(f"[INFO] method                  : {args.reconstruction_method}")
    print_debug(f"[INFO] proj range              : {args.proj_start}:{args.proj_stop}")
    print_debug(f"[INFO] det_x range             : {args.det_x_start}:{args.det_x_stop}")
    print_debug(f"[INFO] det_z range             : {args.det_z_start}:{args.det_z_stop}")
    print_debug(f"[INFO] z_center_mode           : {z_center_mode}")
    print_debug(f"[INFO] auto_z_center           : {args.auto_z_center}")
    print_debug(f"[INFO] auto_z_center_from_angle: {args.auto_z_center_from_angle}")
    print_debug(f"[INFO] z_center_input          : {args.z_center}")
    print_debug(f"[INFO] z_half_width_mm         : {args.z_half_width_mm}")
    print_debug(f"[INFO] z0_fit_mm               : {args.z0_fit_mm}")
    print_debug(f"[INFO] z_center_effective      : {effective_z_center}")
    print_debug(f"[INFO] pitch_mm_per_turn       : {args.pitch_mm_per_turn}")
    print_debug("=" * 70)

    basic_sanity_checks(angles, axial_positions, sinogram, metadata)

    geom_result = build_geometry_v2(
        sinogram_shape=sinogram.shape,
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        proj_start=args.proj_start,
        proj_stop=args.proj_stop,
        det_x_start=args.det_x_start,
        det_x_stop=args.det_x_stop,
        det_z_start=args.det_z_start,
        det_z_stop=args.det_z_stop,
        z_center_mm=effective_z_center,
        z_half_width_mm=args.z_half_width_mm,
        pitch_mm_per_turn=args.pitch_mm_per_turn,
        debug=True,
    )

    sino_crop = crop_sinogram(
        sinogram=sinogram,
        proj_start=args.proj_start,
        proj_stop=args.proj_stop,
        det_x_start=args.det_x_start,
        det_x_stop=args.det_x_stop,
        det_z_start=args.det_z_start,
        det_z_stop=args.det_z_stop,
    )

    expected_shape = tuple(geom_result.cropped_sinogram_shape)
    if sino_crop.shape != expected_shape:
        raise ValueError(f"Cropped sinogram shape mismatch: got {sino_crop.shape}, expected {expected_shape}")

    stem = build_output_stem(
        dataset_name=data_dir.name,
        method=args.reconstruction_method,
        proj_start=args.proj_start,
        proj_stop=args.proj_stop,
        pitch_mm_per_turn=args.pitch_mm_per_turn,
        z_center_mode=z_center_mode,
        z_half_width_mm=args.z_half_width_mm,
    )

    if args.dry_run:
        report = {
            "status": "dry_run_ok",
            "data_dir": str(data_dir),
            "metadata_path": str(args.metadata_path),
            "sinogram_shape_full": list(sinogram.shape),
            "sinogram_shape_crop": list(sino_crop.shape),
            "z_center_mode": z_center_mode,
            "z_center_effective": effective_z_center,
            "z_half_width_mm": args.z_half_width_mm,
            "geometry_debug": geom_result.debug_info,
        }
        report_path = output_dir / f"{stem}_dry_run_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print_debug(f"[OK] Dry-run report saved to {report_path}")
        return

    ray_trafo = geom_result.ray_trafo
    sinogram_element = ray_trafo.range.element(sino_crop)

    t0 = time.time()

    if args.reconstruction_method == "adjoint":
        print_debug("[RUN] Applying adjoint ...")
        reco = ray_trafo.adjoint(sinogram_element)

    elif args.reconstruction_method == "fbp":
        print_debug("[RUN] Building FBP operator ...")
        try:
            fbp = odl.tomo.fbp_op(
                ray_trafo,
                filter_type=args.filter_type,
                frequency_scaling=args.frequency_scaling,
                padding=args.padding,
            )
        except Exception as e:
            raise RuntimeError(
                f"FBP operator construction failed.\n"
                f"Original error: {e}"
            )

        print_debug("[RUN] Applying FBP ...")
        reco = fbp(sinogram_element)

    else:
        raise ValueError(f"Unsupported method: {args.reconstruction_method}")

    dt = time.time() - t0
    print_debug(f"[OK] Reconstruction finished in {dt:.2f} s")

    reco_np = reco.asarray()

    if not np.all(np.isfinite(reco_np)):
        raise ValueError("Reconstruction contains non-finite values.")

    print_debug("[SANITY] Reconstruction stats")
    print_debug(f"[SANITY] reco.shape             : {reco_np.shape}")
    print_debug(f"[SANITY] reco.dtype             : {reco_np.dtype}")
    print_debug(
        f"[SANITY] reco min/max/mean      : "
        f"{float(reco_np.min()):.6f} / {float(reco_np.max()):.6f} / {float(reco_np.mean()):.6f}"
    )

    out_path = output_dir / f"{stem}.nrrd"
    voxel_sizes = geom_result.reco_space.cell_sides
    write_nrrd(out_path, reco_np, voxel_sizes_mm=voxel_sizes)
    print_debug(f"[OK] NRRD saved to {out_path}")

    if args.save_png:
        png_prefix = output_dir / stem
        save_middle_slices_png(reco_np, png_prefix)
        print_debug(f"[OK] Middle-slice PNGs saved with prefix {png_prefix}")

    report = {
        "status": "ok",
        "data_dir": str(data_dir),
        "metadata_path": str(args.metadata_path),
        "method": args.reconstruction_method,
        "filter_type": args.filter_type,
        "frequency_scaling": args.frequency_scaling,
        "padding": args.padding,
        "sinogram_shape_full": list(sinogram.shape),
        "sinogram_shape_crop": list(sino_crop.shape),
        "z_center_mode": z_center_mode,
        "z_center_effective": effective_z_center,
        "z_half_width_mm": args.z_half_width_mm,
        "reconstruction_shape": list(reco_np.shape),
        "reconstruction_stats": {
            "min": float(reco_np.min()),
            "max": float(reco_np.max()),
            "mean": float(reco_np.mean()),
        },
        "runtime_seconds": dt,
        "geometry_debug": geom_result.debug_info,
        "output_nrrd": str(out_path),
    }
    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_debug(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    main()