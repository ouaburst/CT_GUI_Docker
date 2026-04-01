# Script filename: odl_utils_v4.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import numpy as np
import odl


@dataclass
class GlobalGeometryResult:
    ray_trafo: Any
    ray_trafo_adjoint: Any
    reco_space: Any
    geometry: Any
    debug_info: Dict[str, Any]


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def print_debug(msg: str) -> None:
    print(msg, flush=True)


def validate_metadata(metadata: Dict[str, Any]) -> None:
    required = [
        "DET_NPX_Z", "DET_NPX_X", "DET_PIX_SIZE",
        "DET_X_MIN", "DET_X_MAX", "DET_Z_MIN", "DET_Z_MAX",
        "SRC_RADIUS", "DET_RADIUS", "DET_CURVATURE_RADIUS",
    ]
    missing = [k for k in required if k not in metadata]
    if missing:
        raise KeyError(f"Missing keys in metadata: {missing}")


def compute_pitch_mm_per_turn(angles: np.ndarray, axial_positions: np.ndarray) -> float:
    a = np.unwrap(angles.astype(np.float64))
    z = axial_positions.astype(np.float64)
    dtheta = float(a[-1] - a[0])
    dz = float(z[-1] - z[0])
    turns = dtheta / (2.0 * np.pi)
    if abs(turns) < 1e-12:
        raise ValueError("Cannot estimate pitch: angular span too small")
    return float(dz / turns)


def build_global_helical_geometry_v4(
    sinogram_shape: Tuple[int, int, int],
    angles: np.ndarray,
    axial_positions: np.ndarray,
    shifts: np.ndarray,
    metadata: Dict[str, Any],
    rec_min_x: float,
    rec_max_x: float,
    rec_min_y: float,
    rec_max_y: float,
    rec_min_z: float,
    rec_max_z: float,
    rec_voxel_size: float = 1.0,
    use_curved_detector: bool = True,
    use_src_shifts: bool = True,
    use_det_shifts: bool = True,
    debug: bool = True,
) -> GlobalGeometryResult:
    validate_metadata(metadata)

    if len(sinogram_shape) != 3:
        raise ValueError(f"Expected sinogram shape (n_proj, det_x, det_z), got {sinogram_shape}")

    n_proj, _, _ = sinogram_shape

    if len(angles) != n_proj:
        raise ValueError("angles length mismatch")
    if len(axial_positions) != n_proj:
        raise ValueError("axial_positions length mismatch")
    if shifts.shape != (n_proj, 3):
        raise ValueError(f"Expected shifts shape {(n_proj, 3)}, got {shifts.shape}")

    det_x_min = float(metadata["DET_X_MIN"])
    det_x_max = float(metadata["DET_X_MAX"])
    det_z_min = float(metadata["DET_Z_MIN"])
    det_z_max = float(metadata["DET_Z_MAX"])

    detector_partition = odl.uniform_partition(
        min_pt=[det_x_min, det_z_min],
        max_pt=[det_x_max, det_z_max],
        shape=(int(metadata["DET_NPX_X"]), int(metadata["DET_NPX_Z"])),
    )

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

    a = np.unwrap(angles.astype(np.float64))
    a_rel = a - a[0]
    angle_partition = odl.nonuniform_partition(a_rel)

    pitch_mm_per_turn = compute_pitch_mm_per_turn(angles, axial_positions)

    src_radius = float(metadata["SRC_RADIUS"])
    det_radius = float(metadata["DET_RADIUS"])
    det_curvature_radius = float(metadata["DET_CURVATURE_RADIUS"])

    det_curv = (det_curvature_radius, None) if use_curved_detector else None

    shifts64 = shifts.astype(np.float64)

    if use_src_shifts:
        src_shift_func = lambda ang: odl.tomo.flying_focal_spot(ang, apart=angle_partition, shifts=shifts64)
    else:
        src_shift_func = None

    if use_det_shifts:
        det_shift_func = lambda ang: odl.tomo.flying_focal_spot(ang, apart=angle_partition, shifts=shifts64)
    else:
        det_shift_func = None

    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=src_radius,
        det_radius=det_radius,
        det_curvature_radius=det_curv,
        pitch=pitch_mm_per_turn,
        src_shift_func=src_shift_func,
        det_shift_func=det_shift_func,
    )

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    ray_trafo_adjoint = ray_trafo.adjoint

    debug_info = {
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
        "pitch_mm_per_turn": float(pitch_mm_per_turn),
        "src_radius_mm": src_radius,
        "det_radius_mm": det_radius,
        "det_curvature_radius_mm": det_curvature_radius,
        "use_curved_detector": bool(use_curved_detector),
        "use_src_shifts": bool(use_src_shifts),
        "use_det_shifts": bool(use_det_shifts),
        "shifts_stats": {
            "min": shifts64.min(axis=0).tolist(),
            "max": shifts64.max(axis=0).tolist(),
            "mean": shifts64.mean(axis=0).tolist(),
        },
        "mode": "global_helical_geometry_v4",
    }

    if debug:
        print_debug("=" * 70)
        print_debug("[DEBUG] Global geometry summary")
        print_debug(f"[DEBUG] Sinogram shape            : {sinogram_shape}")
        print_debug(f"[DEBUG] Reco shape                : {(nx, ny, nz)}")
        print_debug(f"[DEBUG] Reco voxel size [mm]      : {tuple(float(v) for v in reco_space.cell_sides)}")
        print_debug(f"[DEBUG] Reco X bounds [mm]        : {rec_min_x:.3f} .. {rec_max_x_adj:.3f}")
        print_debug(f"[DEBUG] Reco Y bounds [mm]        : {rec_min_y:.3f} .. {rec_max_y_adj:.3f}")
        print_debug(f"[DEBUG] Reco Z bounds [mm]        : {rec_min_z:.3f} .. {rec_max_z_adj:.3f}")
        print_debug(f"[DEBUG] Angle range rel [rad]     : {float(a_rel[0]):.6f} .. {float(a_rel[-1]):.6f}")
        print_debug(f"[DEBUG] Axial range [mm]          : {float(np.min(axial_positions)):.6f} .. {float(np.max(axial_positions)):.6f}")
        print_debug(f"[DEBUG] Pitch [mm/turn]           : {pitch_mm_per_turn:.6f}")
        print_debug(f"[DEBUG] Curved detector           : {use_curved_detector}")
        print_debug(f"[DEBUG] Source shifts enabled     : {use_src_shifts}")
        print_debug(f"[DEBUG] Detector shifts enabled   : {use_det_shifts}")
        print_debug(f"[DEBUG] shifts min xyz [mm]       : {shifts64.min(axis=0)}")
        print_debug(f"[DEBUG] shifts max xyz [mm]       : {shifts64.max(axis=0)}")
        print_debug("=" * 70)

    return GlobalGeometryResult(
        ray_trafo=ray_trafo,
        ray_trafo_adjoint=ray_trafo_adjoint,
        reco_space=reco_space,
        geometry=geometry,
        debug_info=debug_info,
    )