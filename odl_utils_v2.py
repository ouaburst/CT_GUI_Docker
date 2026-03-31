# Script filename: odl_utils_v2.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import json
import numpy as np
import odl


@dataclass
class GeometryBuildResult:
    ray_trafo: Any
    ray_trafo_adjoint: Any
    reco_space: Any
    geometry: Any
    cropped_angles: np.ndarray
    cropped_axial_positions: np.ndarray
    cropped_sinogram_shape: Tuple[int, int, int]
    debug_info: Dict[str, Any]


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def print_debug(msg: str) -> None:
    print(msg, flush=True)


def _require_keys(metadata: Dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in metadata]
    if missing:
        raise KeyError(f"Missing keys in metadata: {missing}")


def validate_metadata(metadata: Dict[str, Any]) -> None:
    required = [
        "DET_NPX_Z", "DET_NPX_X", "DET_PIX_SIZE",
        "DET_X_MIN", "DET_X_MAX", "DET_Z_MIN", "DET_Z_MAX",
        "SRC_RADIUS", "DET_RADIUS", "DET_CURVATURE_RADIUS",
        "REC_VOXEL_SIZE", "REC_MIN_X", "REC_MAX_X", "REC_MIN_Y", "REC_MAX_Y"
    ]
    _require_keys(metadata, required)

    if metadata["DET_NPX_X"] <= 0 or metadata["DET_NPX_Z"] <= 0:
        raise ValueError("Detector pixel counts must be > 0.")
    if metadata["DET_PIX_SIZE"] <= 0:
        raise ValueError("DET_PIX_SIZE must be > 0.")
    if metadata["REC_VOXEL_SIZE"] <= 0:
        raise ValueError("REC_VOXEL_SIZE must be > 0.")
    if metadata["REC_MAX_X"] <= metadata["REC_MIN_X"]:
        raise ValueError("REC_MAX_X must be > REC_MIN_X.")
    if metadata["REC_MAX_Y"] <= metadata["REC_MIN_Y"]:
        raise ValueError("REC_MAX_Y must be > REC_MIN_Y.")


def _slice_bounds(n: int, start: Optional[int], stop: Optional[int]) -> Tuple[int, int]:
    s = 0 if start is None else int(start)
    e = n if stop is None else int(stop)

    if not (0 <= s < n):
        raise ValueError(f"Invalid slice start={s} for length={n}")
    if not (0 < e <= n):
        raise ValueError(f"Invalid slice stop={e} for length={n}")
    if s >= e:
        raise ValueError(f"Slice start must be < stop, got start={s}, stop={e}")
    return s, e


def _derive_detector_bounds(
    det_min: float,
    pixel_size: float,
    start: int,
    stop: int,
) -> Tuple[float, float]:
    new_min = det_min + start * pixel_size
    new_max = det_min + stop * pixel_size
    return float(new_min), float(new_max)


def _derive_axis_shape_and_max(min_pt: float, max_pt: float, voxel: float) -> Tuple[int, float]:
    length = float(max_pt - min_pt)
    n = int(np.ceil(length / voxel))
    n = max(n, 1)
    max_adjusted = float(min_pt + n * voxel)
    return n, max_adjusted


def estimate_window_z_center_mm(
    axial_positions_full: np.ndarray,
    proj_start: int,
    proj_stop: int,
) -> float:
    if proj_stop <= proj_start:
        raise ValueError("proj_stop must be greater than proj_start")

    mid_idx = (proj_start + proj_stop) // 2
    if not (0 <= mid_idx < len(axial_positions_full)):
        raise ValueError(
            f"Mid index {mid_idx} outside valid range [0, {len(axial_positions_full)-1}]"
        )

    return float(axial_positions_full[mid_idx])


def estimate_window_angle_based_z_center_mm(
    angles_full: np.ndarray,
    proj_start: int,
    proj_stop: int,
    pitch_mm_per_turn: float,
    z0_fit_mm: float,
) -> float:
    if proj_stop <= proj_start:
        raise ValueError("proj_stop must be greater than proj_start")

    angles_win = np.unwrap(angles_full[proj_start:proj_stop].astype(np.float64))
    if len(angles_win) == 0:
        raise ValueError("Selected angle window is empty")

    theta_mid = 0.5 * float(angles_win[0] + angles_win[-1])
    k = float(pitch_mm_per_turn) / (2.0 * np.pi)
    z_center = float(z0_fit_mm + k * theta_mid)
    return z_center


def build_geometry_v2(
    sinogram_shape: Tuple[int, int, int],
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: Dict[str, Any],
    proj_start: Optional[int] = None,
    proj_stop: Optional[int] = None,
    det_x_start: Optional[int] = None,
    det_x_stop: Optional[int] = None,
    det_z_start: Optional[int] = None,
    det_z_stop: Optional[int] = None,
    z_center_mm: float = 0.0,
    z_half_width_mm: float = 10.0,
    pitch_mm_per_turn: float = 40.0,
    debug: bool = True,
) -> GeometryBuildResult:
    """
    Diagnostic version:
    - sinogram shape = (num_proj, det_x, det_z)
    - curved detector enabled
    - manual pitch enabled
    - no src/det shift functions
    - reconstruction z extent forced to a slab centered at z_center_mm
      with half-width z_half_width_mm
    """
    validate_metadata(metadata)

    if len(sinogram_shape) != 3:
        raise ValueError(f"Expected 3D sinogram shape, got {sinogram_shape}")

    n_proj, n_det_x, n_det_z = sinogram_shape

    if len(angles) != n_proj:
        raise ValueError(f"angles length ({len(angles)}) != sinogram projections ({n_proj})")
    if len(axial_positions) != n_proj:
        raise ValueError(f"axial_positions length ({len(axial_positions)}) != sinogram projections ({n_proj})")

    if z_half_width_mm <= 0:
        raise ValueError("z_half_width_mm must be > 0")

    px = float(metadata["DET_PIX_SIZE"])

    p0, p1 = _slice_bounds(n_proj, proj_start, proj_stop)
    x0, x1 = _slice_bounds(n_det_x, det_x_start, det_x_stop)
    z0, z1 = _slice_bounds(n_det_z, det_z_start, det_z_stop)

    angles_c = np.unwrap(angles[p0:p1].astype(np.float64).copy())
    axial_c = axial_positions[p0:p1].astype(np.float64).copy()

    det_x_min, det_x_max = _derive_detector_bounds(float(metadata["DET_X_MIN"]), px, x0, x1)
    det_z_min, det_z_max = _derive_detector_bounds(float(metadata["DET_Z_MIN"]), px, z0, z1)

    detector_partition = odl.uniform_partition(
        min_pt=[det_x_min, det_z_min],
        max_pt=[det_x_max, det_z_max],
        shape=(x1 - x0, z1 - z0),
    )

    voxel = float(metadata["REC_VOXEL_SIZE"])
    rec_min_x = float(metadata["REC_MIN_X"])
    rec_max_x = float(metadata["REC_MAX_X"])
    rec_min_y = float(metadata["REC_MIN_Y"])
    rec_max_y = float(metadata["REC_MAX_Y"])

    z_center = float(z_center_mm)
    rec_min_z = z_center - float(z_half_width_mm)
    rec_max_z = z_center + float(z_half_width_mm)

    nx, rec_max_x_adj = _derive_axis_shape_and_max(rec_min_x, rec_max_x, voxel)
    ny, rec_max_y_adj = _derive_axis_shape_and_max(rec_min_y, rec_max_y, voxel)
    nz, rec_max_z_adj = _derive_axis_shape_and_max(rec_min_z, rec_max_z, voxel)

    reco_space = odl.uniform_discr(
        min_pt=[rec_min_x, rec_min_y, rec_min_z],
        max_pt=[rec_max_x_adj, rec_max_y_adj, rec_max_z_adj],
        shape=[nx, ny, nz],
        dtype="float32",
    )

    angle_partition = odl.nonuniform_partition(angles_c)

    src_radius = float(metadata["SRC_RADIUS"])
    det_radius = float(metadata["DET_RADIUS"])
    det_curvature_radius = float(metadata["DET_CURVATURE_RADIUS"])

    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=src_radius,
        det_radius=det_radius,
        det_curvature_radius=(det_curvature_radius, None),
        pitch=float(pitch_mm_per_turn),
    )

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    ray_trafo_adjoint = ray_trafo.adjoint

    debug_info = {
        "proj_slice": [p0, p1],
        "det_x_slice": [x0, x1],
        "det_z_slice": [z0, z1],
        "cropped_sinogram_shape_expected": [p1 - p0, x1 - x0, z1 - z0],
        "detector_bounds_mm": {
            "x_min": det_x_min,
            "x_max": det_x_max,
            "z_min": det_z_min,
            "z_max": det_z_max,
        },
        "reco_bounds_mm": {
            "x_min": rec_min_x,
            "x_max": rec_max_x_adj,
            "y_min": rec_min_y,
            "y_max": rec_max_y_adj,
            "z_min": rec_min_z,
            "z_max": rec_max_z_adj,
        },
        "reco_shape": [nx, ny, nz],
        "reco_voxel_size_mm": list(map(float, reco_space.cell_sides)),
        "angle_range_rad": [float(angles_c[0]), float(angles_c[-1])],
        "axial_range_mm": [float(axial_c.min()), float(axial_c.max())],
        "src_radius_mm": src_radius,
        "det_radius_mm": det_radius,
        "det_curvature_radius_mm": det_curvature_radius,
        "z_center_mm": z_center,
        "z_half_width_mm": float(z_half_width_mm),
        "pitch_mm_per_turn": float(pitch_mm_per_turn),
        "geometry_mode": "curved_detector_local_helical_slab",
    }

    if debug:
        print_debug("=" * 70)
        print_debug("[DEBUG] Geometry summary")
        print_debug(f"[DEBUG] Input sinogram shape        : {sinogram_shape}")
        print_debug(f"[DEBUG] Cropped sinogram shape      : {(p1 - p0, x1 - x0, z1 - z0)}")
        print_debug(f"[DEBUG] Projection slice           : {p0}:{p1}")
        print_debug(f"[DEBUG] Detector X slice           : {x0}:{x1}")
        print_debug(f"[DEBUG] Detector Z slice           : {z0}:{z1}")
        print_debug(f"[DEBUG] Detector X bounds [mm]     : {det_x_min:.3f} .. {det_x_max:.3f}")
        print_debug(f"[DEBUG] Detector Z bounds [mm]     : {det_z_min:.3f} .. {det_z_max:.3f}")
        print_debug(f"[DEBUG] Reco shape                 : {(nx, ny, nz)}")
        print_debug(f"[DEBUG] Reco voxel size [mm]       : {tuple(float(v) for v in reco_space.cell_sides)}")
        print_debug(f"[DEBUG] Reco X bounds [mm]         : {rec_min_x:.3f} .. {rec_max_x_adj:.3f}")
        print_debug(f"[DEBUG] Reco Y bounds [mm]         : {rec_min_y:.3f} .. {rec_max_y_adj:.3f}")
        print_debug(f"[DEBUG] Reco Z bounds [mm]         : {rec_min_z:.3f} .. {rec_max_z_adj:.3f}")
        print_debug(f"[DEBUG] Z center [mm]             : {z_center:.6f}")
        print_debug(f"[DEBUG] Z half-width [mm]         : {float(z_half_width_mm):.6f}")
        print_debug(f"[DEBUG] Detector curvature [mm]    : {det_curvature_radius:.6f}")
        print_debug(f"[DEBUG] Pitch [mm/turn]           : {float(pitch_mm_per_turn):.6f}")
        print_debug(f"[DEBUG] Geometry mode             : curved_detector_local_helical_slab")
        print_debug("=" * 70)

    return GeometryBuildResult(
        ray_trafo=ray_trafo,
        ray_trafo_adjoint=ray_trafo_adjoint,
        reco_space=reco_space,
        geometry=geometry,
        cropped_angles=angles_c,
        cropped_axial_positions=axial_c,
        cropped_sinogram_shape=(p1 - p0, x1 - x0, z1 - z0),
        debug_info=debug_info,
    )