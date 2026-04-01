# Script filename: reconstruct_full_helical_v1.py
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import nrrd
import numpy as np
import odl

from odl_utils_v2 import (
    load_json,
    build_geometry_v2,
    estimate_window_angle_based_z_center_mm,
    print_debug,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Full helical reconstruction by overlapping local slab FBP fusion"
    )

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--metadata_path", type=str, default="metadata_v2.json")
    ap.add_argument("--output_dir", type=str, default="reconstruction_full_v3")

    ap.add_argument("--window_size", type=int, default=400)
    ap.add_argument("--window_step", type=int, default=200)

    ap.add_argument("--pitch_mm_per_turn", type=float, default=106.5609)
    ap.add_argument("--z0_fit_mm", type=float, default=-2.063)
    ap.add_argument("--z_half_width_mm", type=float, default=10.0)

    ap.add_argument("--filter_type", type=str, default="Ram-Lak")
    ap.add_argument("--frequency_scaling", type=float, default=1.0)
    ap.add_argument("--padding", action="store_true")

    ap.add_argument("--det_x_start", type=int, default=None)
    ap.add_argument("--det_x_stop", type=int, default=None)
    ap.add_argument("--det_z_start", type=int, default=None)
    ap.add_argument("--det_z_stop", type=int, default=None)

    ap.add_argument("--save_window_nrrds", action="store_true")
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
    return np.ascontiguousarray(
        sinogram[proj_start:proj_stop, x0:x1, z0:z1],
        dtype=np.float32,
    )


def make_windows(n_proj: int, window_size: int, window_step: int) -> List[Tuple[int, int]]:
    if window_size <= 0 or window_step <= 0:
        raise ValueError("window_size and window_step must be > 0")
    if window_size > n_proj:
        raise ValueError("window_size cannot exceed number of projections")

    windows = []
    start = 0
    while start + window_size <= n_proj:
        windows.append((start, start + window_size))
        start += window_step

    if not windows:
        windows.append((0, n_proj))
    elif windows[-1][1] < n_proj:
        windows.append((n_proj - window_size, n_proj))

    dedup = []
    seen = set()
    for w in windows:
        if w not in seen:
            dedup.append(w)
            seen.add(w)
    return dedup


def float_tag(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}".replace("-", "m").replace(".", "p")


def unwrap_z_centers(z_centers_wrapped: List[float], period: float) -> List[float]:
    if not z_centers_wrapped:
        return []

    out = [float(z_centers_wrapped[0])]
    offset = 0.0
    prev = float(z_centers_wrapped[0])

    for z in z_centers_wrapped[1:]:
        z = float(z)
        if z - prev < -0.5 * period:
            offset += period
        elif z - prev > 0.5 * period:
            offset -= period
        out.append(z + offset)
        prev = z

    return out


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(args.metadata_path)

    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    sinogram = np.load(data_dir / "sinogram.npy")

    n_proj = sinogram.shape[0]
    windows = make_windows(n_proj, args.window_size, args.window_step)

    print_debug("=" * 80)
    print_debug("[INFO] reconstruct_full_helical_v3.py")
    print_debug(f"[INFO] data_dir            : {data_dir}")
    print_debug(f"[INFO] output_dir         : {output_dir}")
    print_debug(f"[INFO] sinogram shape     : {sinogram.shape}")
    print_debug(f"[INFO] number of windows  : {len(windows)}")
    print_debug(f"[INFO] window size        : {args.window_size}")
    print_debug(f"[INFO] window step        : {args.window_step}")
    print_debug(f"[INFO] pitch_mm_per_turn  : {args.pitch_mm_per_turn}")
    print_debug(f"[INFO] z0_fit_mm          : {args.z0_fit_mm}")
    print_debug(f"[INFO] z_half_width_mm    : {args.z_half_width_mm}")
    print_debug("=" * 80)

    z_centers_wrapped: List[float] = []
    for p0, p1 in windows:
        zc = estimate_window_angle_based_z_center_mm(
            angles_full=angles,
            proj_start=p0,
            proj_stop=p1,
            pitch_mm_per_turn=args.pitch_mm_per_turn,
            z0_fit_mm=args.z0_fit_mm,
        )
        z_centers_wrapped.append(float(zc))

    z_centers_unwrapped = unwrap_z_centers(z_centers_wrapped, args.pitch_mm_per_turn)

    print_debug("[INFO] first wrapped/unwrapped z centers:")
    for i in range(min(10, len(z_centers_wrapped))):
        print_debug(
            f"[INFO]   {i:02d}: wrapped={z_centers_wrapped[i]:8.3f}  "
            f"unwrapped={z_centers_unwrapped[i]:8.3f}"
        )

    slab_infos: List[Dict[str, Any]] = []

    global_min_z = math.inf
    global_max_z = -math.inf
    reference_shape_xy = None
    reference_min_x = None
    reference_min_y = None
    reference_max_x = None
    reference_max_y = None
    voxel_sizes_xyz = None

    t_pass1 = time.time()

    for wi, (p0, p1) in enumerate(windows):
        z_wrapped = z_centers_wrapped[wi]
        z_unwrapped = z_centers_unwrapped[wi]

        # IMPORTANT: geometry uses WRAPPED z center
        geom = build_geometry_v2(
            sinogram_shape=sinogram.shape,
            angles=angles,
            axial_positions=axial_positions,
            metadata=metadata,
            proj_start=p0,
            proj_stop=p1,
            det_x_start=args.det_x_start,
            det_x_stop=args.det_x_stop,
            det_z_start=args.det_z_start,
            det_z_stop=args.det_z_stop,
            z_center_mm=z_wrapped,
            z_half_width_mm=args.z_half_width_mm,
            pitch_mm_per_turn=args.pitch_mm_per_turn,
            debug=False,
        )

        reco_space = geom.reco_space
        min_x, min_y, _ = map(float, reco_space.min_pt)
        max_x, max_y, _ = map(float, reco_space.max_pt)
        sx, sy, sz = map(float, reco_space.cell_sides)
        nx, ny, nz = reco_space.shape

        # IMPORTANT: global placement uses UNWRAPPED z center
        place_min_z = float(z_unwrapped - args.z_half_width_mm)
        place_max_z = float(z_unwrapped + args.z_half_width_mm)

        if reference_shape_xy is None:
            reference_shape_xy = (nx, ny)
            reference_min_x, reference_min_y = min_x, min_y
            reference_max_x, reference_max_y = max_x, max_y
            voxel_sizes_xyz = (sx, sy, sz)
        else:
            if (nx, ny) != reference_shape_xy:
                raise ValueError("Inconsistent XY reconstruction shape across windows")
            if abs(min_x - reference_min_x) > 1e-6 or abs(min_y - reference_min_y) > 1e-6:
                raise ValueError("Inconsistent XY min_pt across windows")
            if abs(max_x - reference_max_x) > 1e-6 or abs(max_y - reference_max_y) > 1e-6:
                raise ValueError("Inconsistent XY max_pt across windows")
            if any(abs(a - b) > 1e-6 for a, b in zip((sx, sy, sz), voxel_sizes_xyz)):
                raise ValueError("Inconsistent voxel sizes across windows")

        global_min_z = min(global_min_z, place_min_z)
        global_max_z = max(global_max_z, place_max_z)

        slab_infos.append(
            {
                "window_index": wi,
                "proj_start": p0,
                "proj_stop": p1,
                "z_center_wrapped_mm": z_wrapped,
                "z_center_unwrapped_mm": z_unwrapped,
                "place_min_z": place_min_z,
                "place_max_z": place_max_z,
                "geom_min_pt": [min_x, min_y, float(reco_space.min_pt[2])],
                "geom_max_pt": [max_x, max_y, float(reco_space.max_pt[2])],
                "shape": [nx, ny, nz],
                "voxel_size_mm": [sx, sy, sz],
            }
        )

    print_debug(f"[INFO] pass1 finished in {time.time() - t_pass1:.2f} s")
    print_debug(f"[INFO] global z range [mm] : {global_min_z:.3f} .. {global_max_z:.3f}")

    sx, sy, sz = voxel_sizes_xyz
    nx, ny = reference_shape_xy
    global_nz = int(np.ceil((global_max_z - global_min_z) / sz))
    global_max_z_adj = global_min_z + global_nz * sz

    fused_sum = np.zeros((nx, ny, global_nz), dtype=np.float32)
    fused_wgt = np.zeros((nx, ny, global_nz), dtype=np.float32)

    manifest_windows: List[Dict[str, Any]] = []
    t_pass2 = time.time()

    for info in slab_infos:
        wi = info["window_index"]
        p0 = info["proj_start"]
        p1 = info["proj_stop"]
        z_wrapped = info["z_center_wrapped_mm"]
        z_unwrapped = info["z_center_unwrapped_mm"]

        print_debug("-" * 80)
        print_debug(
            f"[RUN] window {wi+1}/{len(slab_infos)} : proj {p0}:{p1}, "
            f"z_wrapped={z_wrapped:.3f}, z_unwrapped={z_unwrapped:.3f}"
        )

        # IMPORTANT: geometry still uses WRAPPED z center
        geom = build_geometry_v2(
            sinogram_shape=sinogram.shape,
            angles=angles,
            axial_positions=axial_positions,
            metadata=metadata,
            proj_start=p0,
            proj_stop=p1,
            det_x_start=args.det_x_start,
            det_x_stop=args.det_x_stop,
            det_z_start=args.det_z_start,
            det_z_stop=args.det_z_stop,
            z_center_mm=z_wrapped,
            z_half_width_mm=args.z_half_width_mm,
            pitch_mm_per_turn=args.pitch_mm_per_turn,
            debug=False,
        )

        sino_crop = crop_sinogram(
            sinogram=sinogram,
            proj_start=p0,
            proj_stop=p1,
            det_x_start=args.det_x_start,
            det_x_stop=args.det_x_stop,
            det_z_start=args.det_z_start,
            det_z_stop=args.det_z_stop,
        )
        sino_elem = geom.ray_trafo.range.element(sino_crop)

        fbp = odl.tomo.fbp_op(
            geom.ray_trafo,
            filter_type=args.filter_type,
            frequency_scaling=args.frequency_scaling,
            padding=args.padding,
        )
        slab = fbp(sino_elem).asarray().astype(np.float32)

        slab_min = float(np.min(slab))
        slab_max = float(np.max(slab))
        slab_mean = float(np.mean(slab))

        nx_s, ny_s, nz_s = slab.shape

        # IMPORTANT: placement uses UNWRAPPED z center
        place_min_z = info["place_min_z"]
        z_start_idx = int(round((place_min_z - global_min_z) / sz))
        z_end_idx = z_start_idx + nz_s

        if z_start_idx < 0 or z_end_idx > global_nz:
            raise ValueError("Computed global z placement is out of bounds")

        fused_sum[:, :, z_start_idx:z_end_idx] += slab
        fused_wgt[:, :, z_start_idx:z_end_idx] += 1.0

        if args.save_window_nrrds:
            window_name = (
                f"{data_dir.name}_fbp"
                f"_proj{p0}_{p1}"
                f"_pitch{float_tag(args.pitch_mm_per_turn)}"
                f"_zw{float_tag(z_wrapped, 3)}"
                f"_zuw{float_tag(z_unwrapped, 3)}"
                f"_zhalf{float_tag(args.z_half_width_mm, 1)}"
            )
            write_nrrd(
                output_dir / f"{window_name}.nrrd",
                slab,
                voxel_sizes_mm=(sx, sy, sz),
                min_pt_xyz=(reference_min_x, reference_min_y, place_min_z),
            )

        manifest_windows.append(
            {
                "window_index": wi,
                "proj_start": p0,
                "proj_stop": p1,
                "z_center_wrapped_mm": z_wrapped,
                "z_center_unwrapped_mm": z_unwrapped,
                "place_min_z": place_min_z,
                "place_max_z": info["place_max_z"],
                "shape": [nx_s, ny_s, nz_s],
                "global_z_index_start": z_start_idx,
                "global_z_index_end": z_end_idx,
                "slab_stats": {
                    "min": slab_min,
                    "max": slab_max,
                    "mean": slab_mean,
                },
            }
        )

        print_debug(
            f"[OK] slab stats min/max/mean = "
            f"{slab_min:.6f} / {slab_max:.6f} / {slab_mean:.6f}"
        )

    print_debug(f"[INFO] pass2 finished in {time.time() - t_pass2:.2f} s")

    fused = np.zeros_like(fused_sum, dtype=np.float32)
    np.divide(fused_sum, np.maximum(fused_wgt, 1e-8), out=fused)

    fused_min = float(np.min(fused))
    fused_max = float(np.max(fused))
    fused_mean = float(np.mean(fused))
    fused_std = float(np.std(fused))

    print_debug("=" * 80)
    print_debug("[RESULT] fused volume")
    print_debug(f"[RESULT] shape             : {fused.shape}")
    print_debug(f"[RESULT] min/max/mean/std : {fused_min:.6f} / {fused_max:.6f} / {fused_mean:.6f} / {fused_std:.6f}")
    print_debug("=" * 80)

    fused_name = (
        f"{data_dir.name}_full_helical_fbp_fused"
        f"_w{args.window_size}"
        f"_s{args.window_step}"
        f"_pitch{str(args.pitch_mm_per_turn).replace('.', 'p')}"
        f"_zhalf{str(args.z_half_width_mm).replace('.', 'p')}"
    )
    fused_path = output_dir / f"{fused_name}.nrrd"
    weight_path = output_dir / f"{fused_name}_weights.nrrd"
    manifest_path = output_dir / f"{fused_name}_manifest.json"

    write_nrrd(
        fused_path,
        fused,
        voxel_sizes_mm=(sx, sy, sz),
        min_pt_xyz=(reference_min_x, reference_min_y, global_min_z),
    )
    write_nrrd(
        weight_path,
        fused_wgt,
        voxel_sizes_mm=(sx, sy, sz),
        min_pt_xyz=(reference_min_x, reference_min_y, global_min_z),
    )

    manifest = {
        "status": "ok",
        "data_dir": str(data_dir),
        "metadata_path": str(args.metadata_path),
        "window_size": args.window_size,
        "window_step": args.window_step,
        "pitch_mm_per_turn": args.pitch_mm_per_turn,
        "z0_fit_mm": args.z0_fit_mm,
        "z_half_width_mm": args.z_half_width_mm,
        "filter_type": args.filter_type,
        "frequency_scaling": args.frequency_scaling,
        "padding": args.padding,
        "sinogram_shape": list(sinogram.shape),
        "global_volume": {
            "min_pt": [reference_min_x, reference_min_y, global_min_z],
            "max_pt": [reference_max_x, reference_max_y, global_max_z_adj],
            "shape": list(fused.shape),
            "voxel_size_mm": [sx, sy, sz],
            "stats": {
                "min": fused_min,
                "max": fused_max,
                "mean": fused_mean,
                "std": fused_std,
            },
        },
        "num_windows": len(manifest_windows),
        "windows": manifest_windows,
        "output_fused_nrrd": str(fused_path),
        "output_weight_nrrd": str(weight_path),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print_debug(f"[OK] fused volume saved   : {fused_path}")
    print_debug(f"[OK] weight volume saved  : {weight_path}")
    print_debug(f"[OK] manifest saved       : {manifest_path}")


if __name__ == "__main__":
    main()