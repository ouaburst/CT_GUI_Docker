# Script filename: reconstruction_v5.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import odl


def print_debug(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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


def build_geometry(
    sinogram_shape: tuple[int, int, int],
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: dict,
    rec_min_x: float,
    rec_max_x: float,
    rec_min_y: float,
    rec_max_y: float,
    rec_min_z: float,
    rec_max_z: float,
    rec_voxel_size: float,
    use_curved_detector: bool,
    detector_axes_mode: str,
):
    n_proj = sinogram_shape[0]

    a = np.unwrap(angles.astype(np.float64))
    a_rel = a - a[0]
    apart = odl.nonuniform_partition(a_rel)

    dz = float(axial_positions[-1] - axial_positions[0])
    dtheta = float(a[-1] - a[0])
    turns = dtheta / (2.0 * np.pi)
    if abs(turns) < 1e-12:
        raise ValueError("Angular span too small to estimate pitch")
    pitch = dz / turns

    if detector_axes_mode == "normal":
        # sinogram interpreted as (proj, det_x, det_z)
        dpart = odl.uniform_partition(
            min_pt=[float(metadata["DET_X_MIN"]), float(metadata["DET_Z_MIN"])],
            max_pt=[float(metadata["DET_X_MAX"]), float(metadata["DET_Z_MAX"])],
            shape=(int(metadata["DET_NPX_X"]), int(metadata["DET_NPX_Z"])),
        )
    elif detector_axes_mode == "swapped":
        # sinogram interpreted as (proj, det_z, det_x)
        dpart = odl.uniform_partition(
            min_pt=[float(metadata["DET_Z_MIN"]), float(metadata["DET_X_MIN"])],
            max_pt=[float(metadata["DET_Z_MAX"]), float(metadata["DET_X_MAX"])],
            shape=(int(metadata["DET_NPX_Z"]), int(metadata["DET_NPX_X"])),
        )
    else:
        raise ValueError(f"Unknown detector_axes_mode: {detector_axes_mode}")

    nx = int(np.ceil((rec_max_x - rec_min_x) / rec_voxel_size))
    ny = int(np.ceil((rec_max_y - rec_min_y) / rec_voxel_size))
    nz = int(np.ceil((rec_max_z - rec_min_z) / rec_voxel_size))

    rec_max_x_adj = rec_min_x + nx * rec_voxel_size
    rec_max_y_adj = rec_min_y + ny * rec_voxel_size
    rec_max_z_adj = rec_min_z + nz * rec_voxel_size

    reco_space = odl.uniform_discr(
        min_pt=[rec_min_x, rec_min_y, rec_min_z],
        max_pt=[rec_max_x_adj, rec_max_y_adj, rec_max_z_adj],
        shape=[nx, ny, nz],
        dtype="float32",
    )

    det_curv = (float(metadata["DET_CURVATURE_RADIUS"]), None) if use_curved_detector else None

    geometry = odl.tomo.ConeBeamGeometry(
        apart=apart,
        dpart=dpart,
        src_radius=float(metadata["SRC_RADIUS"]),
        det_radius=float(metadata["DET_RADIUS"]),
        det_curvature_radius=det_curv,
        pitch=float(pitch),
    )

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")

    debug_info = {
        "detector_axes_mode": detector_axes_mode,
        "sinogram_shape": list(sinogram_shape),
        "reco_shape": [nx, ny, nz],
        "reco_bounds_mm": {
            "x_min": rec_min_x,
            "x_max": rec_max_x_adj,
            "y_min": rec_min_y,
            "y_max": rec_max_y_adj,
            "z_min": rec_min_z,
            "z_max": rec_max_z_adj,
        },
        "reco_voxel_size_mm": list(map(float, reco_space.cell_sides)),
        "angle_range_rad_rel": [float(a_rel[0]), float(a_rel[-1])],
        "axial_range_mm": [float(np.min(axial_positions)), float(np.max(axial_positions))],
        "pitch_mm_per_turn": float(pitch),
    }

    return ray_trafo, reco_space, debug_info


def run_one(
    name: str,
    sinogram_np: np.ndarray,
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: dict,
    output_dir: Path,
    rec_min_x: float,
    rec_max_x: float,
    rec_min_y: float,
    rec_max_y: float,
    rec_min_z: float,
    rec_max_z: float,
    rec_voxel_size: float,
    use_curved_detector: bool,
    save_png: bool,
):
    print_debug("=" * 72)
    print_debug(f"[RUN] mode = {name}")
    print_debug(f"[RUN] sinogram shape used : {sinogram_np.shape}")

    ray_trafo, reco_space, debug_info = build_geometry(
        sinogram_shape=sinogram_np.shape,
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        rec_min_x=rec_min_x,
        rec_max_x=rec_max_x,
        rec_min_y=rec_min_y,
        rec_max_y=rec_max_y,
        rec_min_z=rec_min_z,
        rec_max_z=rec_max_z,
        rec_voxel_size=rec_voxel_size,
        use_curved_detector=use_curved_detector,
        detector_axes_mode=name,
    )

    print_debug(f"[DEBUG] pitch [mm/turn] : {debug_info['pitch_mm_per_turn']:.6f}")
    print_debug(f"[DEBUG] reco shape       : {tuple(debug_info['reco_shape'])}")

    sino_elem = ray_trafo.range.element(np.ascontiguousarray(sinogram_np, dtype=np.float32))

    t0 = time.time()
    reco = ray_trafo.adjoint(sino_elem)
    dt = time.time() - t0

    reco_np = reco.asarray().astype(np.float32)

    rmin = float(np.min(reco_np))
    rmax = float(np.max(reco_np))
    rmean = float(np.mean(reco_np))
    rstd = float(np.std(reco_np))

    print_debug(f"[OK] adjoint finished in {dt:.2f} s")
    print_debug(f"[SANITY] reco min/max/mean/std : {rmin:.6f} / {rmax:.6f} / {rmean:.6f} / {rstd:.6f}")

    stem = f"pine_16_1_adjoint_axes_{name}"
    out_path = output_dir / f"{stem}.nrrd"

    write_nrrd(
        out_path,
        reco_np,
        voxel_sizes_mm=reco_space.cell_sides,
        min_pt_xyz=reco_space.min_pt,
    )

    if save_png:
        save_middle_slices_png(reco_np, output_dir / stem)

    report = {
        "status": "ok",
        "mode": name,
        "runtime_seconds": dt,
        "reconstruction_shape": list(reco_np.shape),
        "reconstruction_stats": {
            "min": rmin,
            "max": rmax,
            "mean": rmean,
            "std": rstd,
        },
        "geometry_debug": debug_info,
        "output_nrrd": str(out_path),
    }

    with open(output_dir / f"{stem}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print_debug(f"[OK] saved NRRD   : {out_path}")
    print_debug(f"[OK] saved report : {output_dir / f'{stem}_report.json'}")


def main():
    ap = argparse.ArgumentParser(description="Detector-axis ordering diagnostic")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--metadata_path", type=str, default="metadata_v2.json")
    ap.add_argument("--output_dir", type=str, default="reconstruction_v5")
    ap.add_argument("--rec_min_x", type=float, default=-80.0)
    ap.add_argument("--rec_max_x", type=float, default=130.0)
    ap.add_argument("--rec_min_y", type=float, default=-200.0)
    ap.add_argument("--rec_max_y", type=float, default=20.0)
    ap.add_argument("--rec_min_z", type=float, default=90.0)
    ap.add_argument("--rec_max_z", type=float, default=470.0)
    ap.add_argument("--rec_voxel_size", type=float, default=2.0)
    ap.add_argument("--use_curved_detector", action="store_true")
    ap.add_argument("--save_png", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(args.metadata_path)
    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    sinogram = np.load(data_dir / "sinogram.npy")

    print_debug("[INFO] reconstruction_v5.py")
    print_debug(f"[INFO] original sinogram shape : {sinogram.shape}")

    # normal: (proj, det_x, det_z)
    run_one(
        name="normal",
        sinogram_np=sinogram,
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        output_dir=output_dir,
        rec_min_x=args.rec_min_x,
        rec_max_x=args.rec_max_x,
        rec_min_y=args.rec_min_y,
        rec_max_y=args.rec_max_y,
        rec_min_z=args.rec_min_z,
        rec_max_z=args.rec_max_z,
        rec_voxel_size=args.rec_voxel_size,
        use_curved_detector=args.use_curved_detector,
        save_png=args.save_png,
    )

    # swapped: (proj, det_z, det_x)
    sinogram_swapped = np.transpose(sinogram, (0, 2, 1))
    run_one(
        name="swapped",
        sinogram_np=sinogram_swapped,
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        output_dir=output_dir,
        rec_min_x=args.rec_min_x,
        rec_max_x=args.rec_max_x,
        rec_min_y=args.rec_min_y,
        rec_max_y=args.rec_max_y,
        rec_min_z=args.rec_min_z,
        rec_max_z=args.rec_max_z,
        rec_voxel_size=args.rec_voxel_size,
        use_curved_detector=args.use_curved_detector,
        save_png=args.save_png,
    )


if __name__ == "__main__":
    main()