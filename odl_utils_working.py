# Script filename: odl_utils_working.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple
import json

import numpy as np
import odl


@dataclass
class GeometryBundle:
    ray_trafo: Any
    ray_trafo_adjoint: Any
    reco_space: Any
    geometry: Any
    debug_info: Dict[str, Any]


def print_debug(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def compute_z_shifts(
    angles: np.ndarray,
    axial_positions: np.ndarray,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Faithful standalone version of the original logic.

    Returns
    -------
    shifts_xyz : (N, 3) ndarray
        Only z-shift is nonzero.
    pitch_int : int
        Integer pitch used by the original implementation.
    debug_info : dict
        Diagnostics for sanity checking.
    """
    angles = np.asarray(angles, dtype=np.float64).copy()
    axial_positions = np.asarray(axial_positions, dtype=np.float64).copy()

    if angles.ndim != 1 or axial_positions.ndim != 1:
        raise ValueError("angles and axial_positions must be 1D")
    if len(angles) != len(axial_positions):
        raise ValueError("angles and axial_positions must have same length")
    if len(angles) < 2:
        raise ValueError("Need at least 2 projections")

    z_shifts = angles.copy()

    # Original wrap detection
    wrap_idxs = list(np.where(np.diff(angles) < 0)[0] + 1)
    section_indices = [0] + wrap_idxs + [len(z_shifts) + 1]

    pitches = []
    theta_ac = 0.0
    delta_ac = 0.0

    section_summaries = []

    for index in range(len(section_indices) - 1):
        section_start = section_indices[index]
        section_end = section_indices[index + 1]

        # Clamp because original sentinel is len+1
        section_end_clamped = min(section_end, len(angles))
        if section_end_clamped - section_start < 2:
            continue

        section_positions = axial_positions[section_start:section_end_clamped]
        section_angles = angles[section_start:section_end_clamped]

        section_positions = section_positions - section_positions[0]
        section_angles = section_angles - section_angles[0]

        delta_z = float(section_positions[-1] - section_positions[0])
        delta_theta = float(section_angles[-1] - section_angles[0])

        if abs(delta_theta) < 1e-12:
            continue

        delta_ac += delta_z
        theta_ac += delta_theta

        pitch_value = delta_z / delta_theta
        pitches.append(float(pitch_value * 2.0 * np.pi))

        normalised_pitch = section_angles * pitch_value
        z_shifts[section_start:section_end_clamped] = section_positions - normalised_pitch

        section_summaries.append(
            {
                "section_index": index,
                "start": int(section_start),
                "end": int(section_end_clamped),
                "delta_z_mm": delta_z,
                "delta_theta_rad": delta_theta,
                "pitch_mm_per_turn_local": float(pitch_value * 2.0 * np.pi),
            }
        )

    if abs(theta_ac) < 1e-12:
        raise ValueError("Accumulated theta is too small to estimate pitch")

    pitch_float = delta_ac / (theta_ac / (2.0 * np.pi))
    pitch_int = int(pitch_float)

    shifts_xyz = np.transpose(
        np.vstack(
            [
                np.zeros(len(angles), dtype=np.float64),
                np.zeros(len(angles), dtype=np.float64),
                z_shifts.astype(np.float64),
            ]
        )
    )

    debug_info = {
        "num_views": int(len(angles)),
        "num_wraps": int(len(wrap_idxs)),
        "wrap_indices": [int(x) for x in wrap_idxs[:20]],
        "pitch_float_mm_per_turn": float(pitch_float),
        "pitch_int_mm_per_turn": int(pitch_int),
        "z_shift_min_max_mean": [
            float(np.min(shifts_xyz[:, 2])),
            float(np.max(shifts_xyz[:, 2])),
            float(np.mean(shifts_xyz[:, 2])),
        ],
        "sections": section_summaries,
        "local_pitch_min_max": [
            float(np.min(pitches)) if pitches else None,
            float(np.max(pitches)) if pitches else None,
        ],
    }
    return shifts_xyz, pitch_int, debug_info


def build_working_geometry(
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: Dict[str, Any],
    debug: bool = True,
) -> GeometryBundle:
    """
    Faithful standalone version of original parser_ConeBeamGeometry.
    """
    angles = np.asarray(angles, dtype=np.float64).copy()
    axial_positions = np.asarray(axial_positions, dtype=np.float64).copy()

    # Detector partition: keep original convention exactly
    det_x_min = float(metadata["DET_X_MIN"])
    det_x_max = float(metadata["DET_X_MAX"])
    det_npx_x = int(metadata["DET_NPX_X"])

    det_z_min = float(metadata["DET_Z_MIN"])
    det_z_max = float(metadata["DET_Z_MAX"])
    det_npx_z = int(metadata["DET_NPX_Z"])

    detector_partition = odl.uniform_partition(
        [det_x_min, det_z_min],
        [det_x_max, det_z_max],
        (det_npx_x, det_npx_z),
    )

    # Original z repositioning
    axial_positions -= axial_positions[0]
    axial_positions += 230 * 0.3  # keep original hard-coded shift

    rec_min_x = float(metadata["REC_MIN_X"])
    rec_max_x = float(metadata["REC_MAX_X"])
    rec_min_y = float(metadata["REC_MIN_Y"])
    rec_max_y = float(metadata["REC_MAX_Y"])

    rec_min_z = float(axial_positions[0])
    rec_max_z = float(axial_positions[-1])

    rec_npx_x = int(metadata["REC_NPX_X"])
    rec_npx_y = int(metadata["REC_NPX_Y"])
    rec_pic_size = float(metadata["REC_PIC_SIZE"])
    rec_npx_z = int((rec_max_z - rec_min_z) // rec_pic_size)

    reco_space = odl.uniform_discr(
        min_pt=[rec_min_x, rec_min_y, rec_min_z],
        max_pt=[rec_max_x, rec_max_y, rec_max_z],
        shape=[rec_npx_x, rec_npx_y, rec_npx_z],
        dtype="float32",
    )

    src_radius = float(metadata["SRC_RADIUS"])
    det_radius = float(metadata["DET_RADIUS"])
    det_curvature_radius = float(metadata["DET_CURVATURE_RADIUS"])

    shifts_xyz, pitch_int, shift_debug = compute_z_shifts(angles, axial_positions)

    # Original angle partition: np.unwrap
    angle_partition = odl.nonuniform_partition(np.unwrap(angles))

    # Original use of flying focal spot for both source and detector
    src_shift_func = partial(
        odl.tomo.flying_focal_spot,
        apart=angle_partition,
        shifts=shifts_xyz,
    )
    det_shift_func = partial(
        odl.tomo.flying_focal_spot,
        apart=angle_partition,
        shifts=shifts_xyz,
    )

    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=src_radius,
        det_radius=det_radius,
        det_curvature_radius=(det_curvature_radius, None),
        pitch=pitch_int,
        src_shift_func=src_shift_func,
        det_shift_func=det_shift_func,
    )

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    ray_trafo_adjoint = ray_trafo.adjoint

    debug_info = {
        "detector_partition_shape": [det_npx_x, det_npx_z],
        "detector_bounds": {
            "x_min": det_x_min,
            "x_max": det_x_max,
            "z_min": det_z_min,
            "z_max": det_z_max,
        },
        "reco_shape": [rec_npx_x, rec_npx_y, rec_npx_z],
        "reco_bounds": {
            "x_min": rec_min_x,
            "x_max": rec_max_x,
            "y_min": rec_min_y,
            "y_max": rec_max_y,
            "z_min": rec_min_z,
            "z_max": rec_max_z,
        },
        "reco_voxel_size_mm": list(map(float, reco_space.cell_sides)),
        "src_radius_mm": src_radius,
        "det_radius_mm": det_radius,
        "det_curvature_radius_mm": det_curvature_radius,
        "shift_debug": shift_debug,
        "mode": "working_original_style",
    }

    if debug:
        print_debug("=" * 72)
        print_debug("[DEBUG] Working geometry summary")
        print_debug(f"[DEBUG] Detector shape            : {(det_npx_x, det_npx_z)}")
        print_debug(f"[DEBUG] Detector X bounds        : {det_x_min:.12f} .. {det_x_max:.12f}")
        print_debug(f"[DEBUG] Detector Z bounds [mm]   : {det_z_min:.6f} .. {det_z_max:.6f}")
        print_debug(f"[DEBUG] Reco shape               : {(rec_npx_x, rec_npx_y, rec_npx_z)}")
        print_debug(f"[DEBUG] Reco voxel size [mm]     : {tuple(float(v) for v in reco_space.cell_sides)}")
        print_debug(f"[DEBUG] Reco X bounds [mm]       : {rec_min_x:.3f} .. {rec_max_x:.3f}")
        print_debug(f"[DEBUG] Reco Y bounds [mm]       : {rec_min_y:.3f} .. {rec_max_y:.3f}")
        print_debug(f"[DEBUG] Reco Z bounds [mm]       : {rec_min_z:.3f} .. {rec_max_z:.3f}")
        print_debug(f"[DEBUG] SRC radius [mm]          : {src_radius:.6f}")
        print_debug(f"[DEBUG] DET radius [mm]          : {det_radius:.6f}")
        print_debug(f"[DEBUG] DET curvature [mm]       : {det_curvature_radius:.6f}")
        print_debug(f"[DEBUG] Pitch int [mm/turn]      : {shift_debug['pitch_int_mm_per_turn']}")
        print_debug(f"[DEBUG] Pitch float [mm/turn]    : {shift_debug['pitch_float_mm_per_turn']:.6f}")
        print_debug(f"[DEBUG] Z-shift min/max/mean     : {shift_debug['z_shift_min_max_mean']}")
        print_debug("=" * 72)

    return GeometryBundle(
        ray_trafo=ray_trafo,
        ray_trafo_adjoint=ray_trafo_adjoint,
        reco_space=reco_space,
        geometry=geometry,
        debug_info=debug_info,
    )