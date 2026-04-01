# Script filename: trajectory_debug_v1.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import odl


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_geometry(
    angles: np.ndarray,
    axial_positions: np.ndarray,
    metadata: dict,
    use_curved_detector: bool = True,
):
    det_x_min = float(metadata["DET_X_MIN"])
    det_x_max = float(metadata["DET_X_MAX"])
    det_z_min = float(metadata["DET_Z_MIN"])
    det_z_max = float(metadata["DET_Z_MAX"])

    dpart = odl.uniform_partition(
        min_pt=[det_x_min, det_z_min],
        max_pt=[det_x_max, det_z_max],
        shape=(int(metadata["DET_NPX_X"]), int(metadata["DET_NPX_Z"])),
    )

    a = np.unwrap(angles.astype(np.float64))
    a_rel = a - a[0]
    apart = odl.nonuniform_partition(a_rel)

    dz = float(axial_positions[-1] - axial_positions[0])
    dtheta = float(a[-1] - a[0])
    turns = dtheta / (2.0 * np.pi)
    if abs(turns) < 1e-12:
        raise ValueError("Angular span too small to estimate pitch")
    pitch = dz / turns

    det_curv = (float(metadata["DET_CURVATURE_RADIUS"]), None) if use_curved_detector else None

    geom = odl.tomo.ConeBeamGeometry(
        apart=apart,
        dpart=dpart,
        src_radius=float(metadata["SRC_RADIUS"]),
        det_radius=float(metadata["DET_RADIUS"]),
        det_curvature_radius=det_curv,
        pitch=float(pitch),
    )
    return geom, a_rel, float(pitch)


def main():
    ap = argparse.ArgumentParser(description="Inspect global trajectory implied by ODL geometry")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--metadata_path", type=str, default="metadata_v2.json")
    ap.add_argument("--output_dir", type=str, default="trajectory_debug_v1")
    ap.add_argument("--use_curved_detector", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    angles = np.load(data_dir / "angles.npy")
    axial_positions = np.load(data_dir / "axial_positions.npy")
    shifts = np.load(data_dir / "shifts.npy")
    metadata = load_json(args.metadata_path)

    geom, angles_rel, pitch = build_geometry(
        angles=angles,
        axial_positions=axial_positions,
        metadata=metadata,
        use_curved_detector=args.use_curved_detector,
    )

    src_positions = np.asarray(geom.src_position(angles_rel))
    det_positions = np.asarray(geom.det_refpoint(angles_rel))
    axis_vecs = det_positions - src_positions

    src_z = src_positions[:, 2]
    det_z = det_positions[:, 2]

    report = {
        "n_views": int(len(angles)),
        "angle_rel_min_max": [float(np.min(angles_rel)), float(np.max(angles_rel))],
        "axial_positions_min_max": [float(np.min(axial_positions)), float(np.max(axial_positions))],
        "estimated_pitch_mm_per_turn": float(pitch),
        "src_x_min_max": [float(np.min(src_positions[:, 0])), float(np.max(src_positions[:, 0]))],
        "src_y_min_max": [float(np.min(src_positions[:, 1])), float(np.max(src_positions[:, 1]))],
        "src_z_min_max": [float(np.min(src_z)), float(np.max(src_z))],
        "det_x_min_max": [float(np.min(det_positions[:, 0])), float(np.max(det_positions[:, 0]))],
        "det_y_min_max": [float(np.min(det_positions[:, 1])), float(np.max(det_positions[:, 1]))],
        "det_z_min_max": [float(np.min(det_z)), float(np.max(det_z))],
        "src_z_step_min_max": [
            float(np.min(np.diff(src_z))),
            float(np.max(np.diff(src_z))),
        ],
        "det_z_step_min_max": [
            float(np.min(np.diff(det_z))),
            float(np.max(np.diff(det_z))),
        ],
        "shifts_min_xyz": shifts.min(axis=0).tolist(),
        "shifts_max_xyz": shifts.max(axis=0).tolist(),
        "shifts_mean_xyz": shifts.mean(axis=0).tolist(),
    }

    np.save(output_dir / "src_positions.npy", src_positions)
    np.save(output_dir / "det_positions.npy", det_positions)
    np.save(output_dir / "axis_vecs.npy", axis_vecs)

    with open(output_dir / "trajectory_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[OK] saved:", output_dir / "src_positions.npy")
    print("[OK] saved:", output_dir / "det_positions.npy")
    print("[OK] saved:", output_dir / "axis_vecs.npy")
    print("[OK] saved:", output_dir / "trajectory_report.json")
    print("[INFO] estimated_pitch_mm_per_turn:", report["estimated_pitch_mm_per_turn"])
    print("[INFO] src_z_min_max:", report["src_z_min_max"])
    print("[INFO] det_z_min_max:", report["det_z_min_max"])
    print("[INFO] src_z_step_min_max:", report["src_z_step_min_max"])
    print("[INFO] det_z_step_min_max:", report["det_z_step_min_max"])


if __name__ == "__main__":
    main()