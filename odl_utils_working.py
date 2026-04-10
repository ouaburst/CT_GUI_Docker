# Script filename: odl_utils_working.py
from __future__ import annotations

# Standard library
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple
import json

# Third-party libraries
import numpy as np
import odl


@dataclass
class GeometryBundle:
    """
    Small container that groups together all geometry-related objects.

    Fields
    ------
    ray_trafo :
        The forward projection operator A.
        It maps a reconstruction volume -> sinogram.

    ray_trafo_adjoint :
        The adjoint/backprojection operator A^T.
        It maps a sinogram -> reconstruction volume.

    reco_space :
        The ODL reconstruction space.
        It defines:
        - the physical bounds of the reconstruction volume
        - the voxel grid shape
        - the voxel size

    geometry :
        The ODL ConeBeamGeometry object describing:
        - source trajectory
        - detector geometry
        - helical pitch
        - per-view shifts

    debug_info :
        A dictionary with useful geometry diagnostics that can be saved in reports.
    """
    ray_trafo: Any
    ray_trafo_adjoint: Any
    reco_space: Any
    geometry: Any
    debug_info: Dict[str, Any]


def print_debug(msg: str) -> None:
    """
    Print a message immediately.

    Why use this wrapper?
    ---------------------
    flush=True forces the text to appear directly in the terminal,
    which is helpful for long-running reconstructions where you want
    progress messages without waiting for the output buffer.
    """
    print(msg, flush=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file and return it as a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file, for example metadata.json.

    Returns
    -------
    dict
        Parsed JSON content.

    Typical use here
    ----------------
    This function is mainly used to load metadata such as:
    - detector size
    - detector bounds
    - source radius
    - reconstruction bounds
    """
    with open(path, "r") as f:
        return json.load(f)


def compute_z_shifts(
    angles: np.ndarray,
    axial_positions: np.ndarray,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Compute per-view z-shifts and a global integer pitch.

    This is a faithful standalone version of the original logic.

    Inputs
    ------
    angles :
        1D array of projection angles.
        The sequence may contain wraps, i.e. it may restart after one turn.

    axial_positions :
        1D array of measured source/detector axial positions along z.

    Output
    ------
    shifts_xyz : (N, 3) ndarray
        Per-view shift array used by ODL.
        In this implementation:
        - x shift = 0
        - y shift = 0
        - z shift = computed residual correction

    pitch_int : int
        Integer pitch in mm/turn, matching the original implementation.

    debug_info : dict
        Extra diagnostics:
        - number of wraps
        - local pitch values
        - min/max z-shift
        - section summaries

    Core idea
    ---------
    The measured helical trajectory is not perfectly ideal.
    So the code:
    1. splits the acquisition into sections between angle wraps
    2. estimates a local pitch in each section
    3. subtracts the ideal helical trend from measured z
    4. stores the residual as z-shift correction

    Why this is useful
    ------------------
    ODL's ConeBeamGeometry uses:
    - a helical pitch
    - optional per-view source/detector shifts

    This function provides both from the measured acquisition data.
    """
    # Convert to float64 for more stable numerical calculations
    angles = np.asarray(angles, dtype=np.float64).copy()
    axial_positions = np.asarray(axial_positions, dtype=np.float64).copy()

    # Basic input validation
    if angles.ndim != 1 or axial_positions.ndim != 1:
        raise ValueError("angles and axial_positions must be 1D")
    if len(angles) != len(axial_positions):
        raise ValueError("angles and axial_positions must have same length")
    if len(angles) < 2:
        raise ValueError("Need at least 2 projections")

    # Start with a placeholder array.
    # Later, this will store the z-residual shift per view.
    z_shifts = angles.copy()

    # Detect wraps in the angle sequence.
    # A wrap happens where angle suddenly decreases.
    # Example:
    #   ..., 6.20, 6.28, 0.01, 0.03, ...
    wrap_idxs = list(np.where(np.diff(angles) < 0)[0] + 1)

    # Define sections separated by wraps.
    # Original code uses a sentinel len+1, so we keep that style.
    section_indices = [0] + wrap_idxs + [len(z_shifts) + 1]

    # Store local pitch estimates (one per section)
    pitches = []

    # Accumulate global z and angle changes over valid sections
    theta_ac = 0.0
    delta_ac = 0.0

    # Human-readable summaries for debugging/reporting
    section_summaries = []

    # Process each section independently
    for index in range(len(section_indices) - 1):
        section_start = section_indices[index]
        section_end = section_indices[index + 1]

        # Clamp because the sentinel is len+1
        section_end_clamped = min(section_end, len(angles))

        # Skip sections with too few samples
        if section_end_clamped - section_start < 2:
            continue

        # Extract section data
        section_positions = axial_positions[section_start:section_end_clamped]
        section_angles = angles[section_start:section_end_clamped]

        # Normalize section so it starts at zero
        section_positions = section_positions - section_positions[0]
        section_angles = section_angles - section_angles[0]

        # Total z and angular change in this section
        delta_z = float(section_positions[-1] - section_positions[0])
        delta_theta = float(section_angles[-1] - section_angles[0])

        # Skip degenerate cases
        if abs(delta_theta) < 1e-12:
            continue

        # Accumulate global totals for final pitch estimate
        delta_ac += delta_z
        theta_ac += delta_theta

        # Local pitch in mm/rad
        pitch_value = delta_z / delta_theta

        # Convert local pitch to mm/turn and store it
        pitches.append(float(pitch_value * 2.0 * np.pi))

        # Ideal z trajectory in this section assuming constant pitch
        normalised_pitch = section_angles * pitch_value

        # Residual between measured z and ideal helix
        # This residual becomes the per-view z-shift correction
        z_shifts[section_start:section_end_clamped] = section_positions - normalised_pitch

        # Save debug info for this section
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

    # Need a nonzero total angular span to estimate global pitch
    if abs(theta_ac) < 1e-12:
        raise ValueError("Accumulated theta is too small to estimate pitch")

    # Global pitch in mm/turn
    pitch_float = delta_ac / (theta_ac / (2.0 * np.pi))

    # Original implementation keeps an integer pitch
    pitch_int = int(pitch_float)

    # Build ODL shift array of shape (N, 3)
    # Only z-component is nonzero
    shifts_xyz = np.transpose(
        np.vstack(
            [
                np.zeros(len(angles), dtype=np.float64),   # x shifts
                np.zeros(len(angles), dtype=np.float64),   # y shifts
                z_shifts.astype(np.float64),               # z shifts
            ]
        )
    )

    # Collect diagnostics
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
    Build the full ODL reconstruction geometry using the original working convention.

    Inputs
    ------
    angles :
        1D array of projection angles.

    axial_positions :
        1D array of axial positions along z.

    metadata :
        Dictionary loaded from metadata.json.
        It contains:
        - detector bounds
        - detector size
        - source radius
        - detector radius
        - detector curvature radius
        - reconstruction bounds in x and y
        - reconstruction pixel size

    debug :
        If True, print a detailed geometry summary.

    Output
    ------
    GeometryBundle
        Contains:
        - forward projector
        - adjoint projector
        - reconstruction space
        - geometry object
        - debug dictionary

    Main steps
    ----------
    1. Build detector partition using the original convention.
    2. Reposition axial positions in z using the original hard-coded offset.
    3. Build reconstruction space.
    4. Estimate z-shifts and pitch using compute_z_shifts().
    5. Build ODL ConeBeamGeometry with:
       - curved detector
       - pitch
       - source shifts
       - detector shifts
    6. Create CUDA ray transform and adjoint.

    Why this matters
    ----------------
    This function is the key reason the reconstruction works.
    It preserves the original geometry convention that matched your data.
    """
    # Convert to float64 copies so we can modify values safely
    angles = np.asarray(angles, dtype=np.float64).copy()
    axial_positions = np.asarray(axial_positions, dtype=np.float64).copy()

    # ------------------------------------------------------------
    # Detector geometry
    # ------------------------------------------------------------
    # Keep the original convention exactly.
    # Note: DET_X_MIN/MAX are not large mm values here;
    # they match the original curved-detector parameterization.
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

    # ------------------------------------------------------------
    # Original z repositioning
    # ------------------------------------------------------------
    # Move axial positions so they start at zero,
    # then apply the original hard-coded offset:
    #   230 detector rows * 0.3 mm = 69 mm
    axial_positions -= axial_positions[0]
    axial_positions += 230 * 0.3

    # ------------------------------------------------------------
    # Reconstruction bounds in x and y come from metadata
    # Reconstruction bounds in z come from the repositioned axial trajectory
    # ------------------------------------------------------------
    rec_min_x = float(metadata["REC_MIN_X"])
    rec_max_x = float(metadata["REC_MAX_X"])
    rec_min_y = float(metadata["REC_MIN_Y"])
    rec_max_y = float(metadata["REC_MAX_Y"])

    rec_min_z = float(axial_positions[0])
    rec_max_z = float(axial_positions[-1])

    # Reconstruction shape in x/y is fixed in metadata
    rec_npx_x = int(metadata["REC_NPX_X"])
    rec_npx_y = int(metadata["REC_NPX_Y"])

    # Reconstruction z resolution is set by REC_PIC_SIZE
    rec_pic_size = float(metadata["REC_PIC_SIZE"])
    rec_npx_z = int((rec_max_z - rec_min_z) // rec_pic_size)

    # Create reconstruction space
    reco_space = odl.uniform_discr(
        min_pt=[rec_min_x, rec_min_y, rec_min_z],
        max_pt=[rec_max_x, rec_max_y, rec_max_z],
        shape=[rec_npx_x, rec_npx_y, rec_npx_z],
        dtype="float32",
    )

    # ------------------------------------------------------------
    # Scanner geometry parameters
    # ------------------------------------------------------------
    src_radius = float(metadata["SRC_RADIUS"])
    det_radius = float(metadata["DET_RADIUS"])
    det_curvature_radius = float(metadata["DET_CURVATURE_RADIUS"])

    # Estimate per-view z shifts and integer pitch
    shifts_xyz, pitch_int, shift_debug = compute_z_shifts(angles, axial_positions)

    # Build angle partition using unwrapped angles
    angle_partition = odl.nonuniform_partition(np.unwrap(angles))

    # ------------------------------------------------------------
    # Source and detector shift functions
    # ------------------------------------------------------------
    # The original implementation uses flying_focal_spot for both source and detector,
    # with the same measured z-shift array.
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

    # ------------------------------------------------------------
    # Build the full cone-beam geometry
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Build forward and adjoint operators
    # ------------------------------------------------------------
    # ray_trafo         = A
    # ray_trafo.adjoint = A^T
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    ray_trafo_adjoint = ray_trafo.adjoint

    # Save useful diagnostics
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

    # Optional terminal printout for sanity checking
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