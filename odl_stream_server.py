from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import json
import odl
import pyvista as pv
from scipy.interpolate import interp1d
import tempfile
import nrrd
import subprocess
import argparse
import runpy


# #######################################################
# Run:
#   uvicorn odl_stream_server:app --reload --host 0.0.0.0 --port 8000
# Notes:
# - The app loads a default sample on startup from slicer_backend_config.json
# - It streams geometry windows and per-projection sinogram slices
# - It can also run reconstructions by invoking reconstruction.py
# #######################################################


# ---------------------------
# Globals (populated on load)
# ---------------------------
sinogram = None          # numpy array with shape (N, det_x, det_z)
angles = None            # raw angles array (not directly used after sort)
z_positions = None       # raw axial positions (not directly used after correction)
shifts = None            # per-projection shifts (x,y,z)
metadata = None          # dict with geometry + reconstruction grid settings
geometry = None          # ODL ConeBeamGeometry constructed from dataset
N = 0                    # number of projections
pix_size_det = 0.0       # detector pixel size (mm)
cached_full_trajectory = None  # list of 3D source positions for all indices


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Directory where reconstruction outputs (NRRDs) will be saved/served
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "slicer_backend_config.json"
FULL_GEOM_JSON = OUTPUT_DIR / "full_geometry.json"
TRAJECTORY_JSON = OUTPUT_DIR / "full_trajectory.json"


#########################################################
# _load_config
# Load backend config JSON which describes available samples
# and the bind-mounted volume root.
#########################################################
def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return json.load(f)


#########################################################
# _resolve_sample_dir
# Build the absolute path to a sample directory given specie,
# tree_ID and disk_ID, according to the documented layout.
#########################################################
def _resolve_sample_dir(cfg: dict, specie: str, tree_ID: int, disk_ID: int) -> Path:
    """
    Expected layout on host (bind-mount into the container at the SAME path):
      <volume_name>/real_datasets/ml_ready/<specie>_<tree_ID>_<disk_ID>/
        ├─ sinogram.npy
        ├─ angles.npy
        ├─ axial_positions.npy
        ├─ shifts.npy
        └─ metadata.json
    """
    vol_root = Path(cfg["volume_name"]) / "real_datasets" / "ml_ready"
    return vol_root / f"{specie}_{tree_ID}_{disk_ID}"


#########################################################
# _validate_reconstruction_geometry
# Sanity-check the reconstruction volume dimensions against
# metadata-derived extents (REC_MIN/MAX_* and REC_NPX_*).
#########################################################
def _validate_reconstruction_geometry(md: dict):
    npx_x = md["REC_NPX_X"]
    npx_y = md["REC_NPX_Y"]
    pix_size = md["REC_PIC_SIZE"]

    min_x = md["REC_MIN_X"]; max_x = md["REC_MAX_X"]
    min_y = md["REC_MIN_Y"]; max_y = md["REC_MAX_Y"]

    expected_width = (max_x - min_x)
    expected_height = (max_y - min_y)
    actual_width = npx_x * pix_size
    actual_height = npx_y * pix_size

    print("\n[CHECK] Verifying reconstruction volume dimensions...")
    print(f"  → Width: {actual_width:.2f} mm (expected: {expected_width:.2f} mm)")
    print(f"  → Height: {actual_height:.2f} mm (expected: {expected_height:.2f} mm)")
    if abs(actual_width - expected_width) > 1e-2 or abs(actual_height - expected_height) > 1e-2:
        print("  ! Dimension mismatch!")
    else:
        print("  ✓ Volume size matches metadata.")
    cx = (min_x + max_x) / 2; cy = (min_y + max_y) / 2
    print(f"  Center of volume: ({cx:.2f}, {cy:.2f})")


#########################################################
# _load_sample_data
# Load arrays from disk and construct an ODL ConeBeamGeometry.
# - Applies z-shift correction and angle sort
# - Estimates helical pitch
# - Builds angle & detector partitions
# - Stores into module-level globals
#########################################################
def _load_sample_data(sample_dir: Path):
    global sinogram, angles, z_positions, shifts, metadata, geometry, N, pix_size_det

    print(f"[INFO] Loading data from: {sample_dir}")
    req = ["sinogram.npy", "angles.npy", "axial_positions.npy", "shifts.npy", "metadata.json"]
    missing = [p for p in req if not (sample_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {sample_dir}: {missing}")

    # Load raw arrays
    sinogram = np.load(sample_dir / "sinogram.npy")                 # expected (N, det_x, det_z)
    angles_raw = np.mod(np.load(sample_dir / "angles.npy"), 2*np.pi)
    z_raw = np.load(sample_dir / "axial_positions.npy")
    shifts_raw = np.load(sample_dir / "shifts.npy")
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"[INFO] Sinogram loaded with shape {sinogram.shape}")

    # Apply z correction (accounting for any per-view shift in z)
    z_corrected = z_raw + shifts_raw[:, 2]

    # Sort by angle to ensure monotonic progression along the scan
    sorted_indices = np.argsort(angles_raw)
    angles_sorted = angles_raw[sorted_indices]
    z_sorted = z_corrected[sorted_indices]

    # Cache counts and detector pixel size
    N = len(angles_sorted)
    pix_size_det = float(metadata["DET_PIX_SIZE"])

    # Estimate helical pitch (z advance per full 2π revolution)
    angle_range = float(angles_sorted[-1] - angles_sorted[0])
    z_range = float(z_sorted[-1] - z_sorted[0])
    pitch = z_range * (2 * np.pi) / angle_range if angle_range != 0 else 0.0
    print(f"[INFO] Computed pitch: {pitch:.2f} mm")

    # Create ODL angle partition over [0, 2π) of length N
    angle_partition = odl.uniform_partition(0, 2 * np.pi, N)

    # Create 2D detector partition using physical extents and number of pixels
    detector_partition = odl.uniform_partition(
        [metadata["DET_X_MIN"], metadata["DET_Z_MIN"]],
        [metadata["DET_X_MAX"], metadata["DET_Z_MAX"]],
        [metadata["DET_NPX_X"], metadata["DET_NPX_Z"]],
    )

    # Keep an angle→z interpolation (currently not plugged into src_shift_func)
    _ = interp1d(
        angles_sorted, z_sorted, kind="linear",
        bounds_error=False, fill_value=(z_sorted[0], z_sorted[-1])
    )

    # Build helical cone-beam geometry; det_curvature_radius[0] is tangential curvature
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=metadata["SRC_RADIUS"],
        det_radius=metadata["DET_RADIUS"],
        det_curvature_radius=[metadata["DET_CURVATURE_RADIUS"], None],
        pitch=pitch,
        axis=[0, 0, 1],
        src_shift_func=None,     # could be set to a function of angle if needed
        translation=[0, 0, 0],
    )

    # Optional sanity checks against reconstruction grid definition
    _validate_reconstruction_geometry(metadata)


#########################################################
# get_bezier_surface_points
# Approximate the curved detector panel as a quadratic Bézier
# surface mesh for visualization (num_v × num_u control).
#########################################################
def get_bezier_surface_points(i, num_u=8, num_v=4):
    """Return (num_v x num_u x 3) detector surface points for index i using a quadratic Bézier grid."""
    angle = geometry.angles[i]
    src = np.asarray(geometry.src_position(angle)).reshape(3)
    det_ctr = np.asarray(geometry.det_refpoint(angle)).reshape(3)
    du, dv = geometry.det_axes(angle)
    du = np.asarray(du).reshape(3); dv = np.asarray(dv).reshape(3)

    # Determine tangential curvature range from pixel count and pixel size
    R_curv = float(metadata["DET_CURVATURE_RADIUS"])
    n_x = int(metadata["DET_NPX_X"])
    arc_length = n_x * pix_size_det
    theta_range = arc_length / R_curv

    # Use 3 control columns in theta (quadratic Bézier) and 3 rows in z
    theta_vals = np.linspace(-theta_range / 2, theta_range / 2, 3)
    z_vals = np.linspace(metadata["DET_Z_MIN"], metadata["DET_Z_MAX"], 3)

    # Build 3×3 control grid in 3D
    control_points = []
    for z in z_vals:
        row = []
        for theta in theta_vals:
            # Outward normal approx from detector center to source
            normal = src - det_ctr
            normal = normal / np.linalg.norm(normal)
            # Arc along du + sagitta along normal + vertical offset along dv
            pt = det_ctr + R_curv * np.sin(theta) * du + R_curv * (1 - np.cos(theta)) * normal + z * dv
            row.append(pt)
        control_points.append(row)
    control_points = np.array(control_points)  # (3, 3, 3)

    # Local helper: 1D quadratic Bézier interpolation for 3 control pts
    def bezier_interp_1d(ctrl, t):
        return (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * ctrl[2]

    # Sample u (tangential) and v (vertical)
    u_vals = np.linspace(0, 1, num_u)
    v_vals = np.linspace(0, 1, num_v)

    # Interpolate a dense surface by Bézier in u then v
    surface_points = []
    for v in v_vals:
        row = []
        interm_rows = [None, None, None]
        for u in u_vals:
            for r in range(3):
                interm_rows[r] = bezier_interp_1d(control_points[r], u)
            pt = bezier_interp_1d(np.array(interm_rows), v)
            row.append(pt.tolist())
        surface_points.append(row)
    return surface_points  # (num_v, num_u, 3)


#########################################################
# get_geometry_at
# Compute a simplified panel mesh + 4 FOV rays and the source
# position for a given index i. If curvature radius > 0, model
# a curved panel arc; else fall back to a flat rectangle.
#########################################################
def get_geometry_at(i, nx=20, nz=2):
    """Return (src point, detector_mesh grid [nz x nx x 3], fov rays list of [p0, p1])."""
    angle = geometry.angles[i]
    src = np.asarray(geometry.src_position(angle)).reshape(3)
    det_ctr = np.asarray(geometry.det_refpoint(angle)).reshape(3)
    du, dv = geometry.det_axes(angle)
    du = np.asarray(du).reshape(3); dv = np.asarray(dv).reshape(3)

    R_curv = float(metadata.get("DET_CURVATURE_RADIUS", 0.0))
    n_x = int(metadata["DET_NPX_X"]); n_z = int(metadata["DET_NPX_Z"])

    if R_curv > 0:
        # Build curved panel sample grid in tangential angle (theta) and detector z
        arc_length = n_x * pix_size_det
        theta_range = arc_length / R_curv
        theta_vals = np.linspace(-theta_range / 2, theta_range / 2, nx)
        z_vals = np.linspace(metadata["DET_Z_MIN"], metadata["DET_Z_MAX"], nz)

        detector_mesh = []
        for z in z_vals:
            row = []
            for theta in theta_vals:
                normal = src - det_ctr
                normal = normal / np.linalg.norm(normal)
                pt = det_ctr + R_curv * np.sin(theta) * du + R_curv * (1 - np.cos(theta)) * normal + z * dv
                row.append(pt.tolist())
            detector_mesh.append(row)

        # 4 representative rays to panel corners (min/max theta × min/max z)
        rays = [
            [src.tolist(), detector_mesh[0][0]],
            [src.tolist(), detector_mesh[0][-1]],
            [src.tolist(), detector_mesh[-1][-1]],
            [src.tolist(), detector_mesh[-1][0]],
        ]
    else:
        # Flat fallback (unlikely for your setup)
        w = n_x * pix_size_det / 2.0
        h = n_z * pix_size_det / 2.0
        corners = [
            (det_ctr + sx * w * du + sz * h * dv).tolist()
            for sx, sz in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ]
        detector_mesh = [[corners[0], corners[1]], [corners[3], corners[2]]]
        rays = [[src.tolist(), c] for c in corners]

    return src.tolist(), detector_mesh, rays


# ----------------------------------
# VTP writers (kept for completeness)
# ----------------------------------

#########################################################
# save_points_vtp
# Save a polyline through the given list of points as VTP.
#########################################################
def save_points_vtp(points, filename):
    points = np.array(points)
    n_points = len(points)
    lines = np.hstack([[n_points] + list(range(n_points))])
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = lines
    mesh.save(str(filename))


#########################################################
# save_quads_vtp
# Save a set of quadrilateral faces given as quads (4 vertices).
#########################################################
def save_quads_vtp(quads, filename):
    mesh = pv.PolyData()
    mesh.points = np.array([pt for quad in quads for pt in quad])
    faces = [[4] + list(range(i * 4, i * 4 + 4)) for i in range(len(quads))]
    mesh.faces = np.hstack(faces)
    mesh.save(str(filename))


#########################################################
# save_lines_vtp
# Save multiple independent line segments as a single VTP.
#########################################################
def save_lines_vtp(lines, filename):
    mesh = pv.PolyData()
    points = []
    cells = []
    for i, (p1, p2) in enumerate(lines):
        points.extend([p1, p2])
        cells.append([2, 2 * i, 2 * i + 1])
    mesh.points = np.array(points)
    mesh.lines = np.hstack(cells)
    mesh.save(str(filename))


# ----------------------------------
# FastAPI app
# ----------------------------------
app = FastAPI()

# Serve static reconstruction outputs (NRRDs produced by run_reconstruction)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


#########################################################
# get_backend_config (route)
# Serve slicer_backend_config.json for legacy clients.
#########################################################
@app.api_route("/slicer_backend_config.json", methods=["GET", "HEAD"])
def get_backend_config():
    if CONFIG_PATH.exists():
        return FileResponse(CONFIG_PATH, media_type="application/json", filename="slicer_backend_config.json")
    return JSONResponse(status_code=404, content={"error": f"Missing config file at {str(CONFIG_PATH)}"})


#########################################################
# serve_file (route)
# Small file server for anything under ./output (e.g., VTP/JSON).
#########################################################
@app.get("/files/{filename}")
def serve_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    # naive media type; override per your needs
    return FileResponse(path, media_type="application/octet-stream")


#########################################################
# stream_window (route)
# Return a local geometry window around index i (±n):
# - sources: list of 3D source points
# - detector_panels: coarse panel meshes
# - fov_rays: corner rays to panel
# - bezier_curves: smooth curved panel surfaces
# Also maintains a cached full source trajectory on disk.
#########################################################
@app.get("/stream_window")
def stream_window(i: int = Query(...), n: int = Query(10), format: str = Query("json")):
    assert 0 <= i < N, "Index out of bounds"

    if n == 0:
        i_start = i
        i_end = i + 1
        print(f"[INFO] Serving single acquisition at index {i}")
    else:
        i_start = max(0, i - n)
        i_end = min(N, i + n + 1)
        print(f"[INFO] Serving geometry: i={i}, window=({i_start}, {i_end}), format={format}")

    sources, all_quads, all_rays, bezier_surfaces = [], [], [], []
    for i_sorted in range(i_start, i_end):
        src, corners, rays = get_geometry_at(i_sorted)
        sources.append(src)
        all_quads.append(corners)
        all_rays.append(rays)
        bezier_surfaces.append(get_bezier_surface_points(i_sorted))

    # Cache or load full trajectory to avoid recompute
    global cached_full_trajectory
    if cached_full_trajectory is None:
        if TRAJECTORY_JSON.exists():
            print(f"[INFO] Loading full trajectory from disk...")
            with open(TRAJECTORY_JSON, "r") as f:
                cached_full_trajectory = json.load(f)
        else:
            print("[INFO] Computing full source trajectory and saving to disk...")
            cached_full_trajectory = []
            for idx in range(N):
                src_pos, _, _ = get_geometry_at(idx)
                cached_full_trajectory.append(src_pos)
            with open(TRAJECTORY_JSON, "w") as f:
                json.dump(cached_full_trajectory, f)
    else:
        print("[INFO] Using cached full source trajectory.")

    return JSONResponse(content={
        "sources": sources,
        "detector_panels": all_quads,
        "fov_rays": all_rays,
        "full_trajectory": cached_full_trajectory,
        "total_angles": N,
        "bezier_curves": bezier_surfaces
    })


#########################################################
# full_dataset (route)
# Send the entire sinogram and per-index geometry payload.
# WARNING: Can be large; intended for debugging/exports,
# not real-time streaming to clients.
#########################################################
@app.get("/full_dataset")
def full_dataset():
    global cached_full_trajectory

    if cached_full_trajectory is None:
        print("[INFO] Computing full source trajectory...")
        cached_full_trajectory = []
        for idx in range(N):
            src_pos, _, _ = get_geometry_at(idx)
            cached_full_trajectory.append(src_pos)
        with open(TRAJECTORY_JSON, "w") as f:
            json.dump(cached_full_trajectory, f)
    else:
        print("[INFO] Using cached full source trajectory.")

    print("[INFO] Preparing full geometry and sinogram for streaming...")

    geometry_data = []
    for i in range(N):
        src, corners, rays = get_geometry_at(i)
        geometry_data.append({
            "index": i,
            "source": src,
            "corners": corners,
            "rays": rays
        })

    print("[INFO] Dataset ready for transmission.")
    return JSONResponse(content={
        "sinogram": sinogram.tolist(),  # shape: (N, det_x, det_z)
        "geometry": geometry_data,      # per-index geometry
        "trajectory": cached_full_trajectory,
        "total_angles": N
    })


#########################################################
# generate_full_geometry (startup)
# On app startup:
# - Load config and the first sample
# - Precompute and persist a compact geometry JSON
# - Persist the full trajectory JSON
#########################################################
@app.on_event("startup")
def generate_full_geometry():
    """Load config + default sample, then write output/full_geometry.json and cache trajectory."""
    global cached_full_trajectory

    cfg = _load_config()
    if not cfg.get("samples"):
        raise RuntimeError("No samples listed in slicer_backend_config.json")
    first = cfg["samples"][0]
    sample_dir = _resolve_sample_dir(cfg, first["specie"], int(first["tree_ID"]), int(first["disk_ID"]))
    _load_sample_data(sample_dir)

    print("[INFO] Precomputing and saving full geometry JSON to disk...")

    data = {
        "sources": [],
        "detector_panels": [],
        "fov_rays": [],
        "full_trajectory": [],
        "bezier_curves": []
    }

    for i in range(N):
        src, corners, rays = get_geometry_at(i)
        bezier = get_bezier_surface_points(i)
        data["sources"].append(src)
        data["detector_panels"].append(corners)
        data["fov_rays"].append(rays)
        data["full_trajectory"].append(src)
        data["bezier_curves"].append(bezier)

    with open(FULL_GEOM_JSON, "w") as f:
        json.dump(data, f)

    cached_full_trajectory = data["full_trajectory"]
    with open(TRAJECTORY_JSON, "w") as f:
        json.dump(cached_full_trajectory, f)

    print(f"[INFO] Full geometry saved to {FULL_GEOM_JSON}")
    print(f"[INFO] Full trajectory saved to {TRAJECTORY_JSON}")


#########################################################
# get_full_geometry (route)
# Serve the compact precomputed full geometry JSON file.
#########################################################
@app.get("/full_geometry")
def get_full_geometry():
    if not FULL_GEOM_JSON.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(FULL_GEOM_JSON, media_type="application/json")


#########################################################
# serve_full_trajectory (route)
# Serve only the full source trajectory JSON.
#########################################################
@app.get("/full_trajectory.json")
def serve_full_trajectory():
    if not TRAJECTORY_JSON.exists():
        return JSONResponse(status_code=404, content={"error": "Trajectory file not found"})
    return FileResponse(TRAJECTORY_JSON, media_type="application/json")


# ----- Switch sample at runtime (optional)
class SampleSelect(BaseModel):
    specie: str
    tree_ID: int
    disk_ID: int


#########################################################
# select_sample (route)
# Switch active dataset (specie/tree/disk), reload arrays,
# and rebuild cached geometry + trajectory JSONs.
#########################################################
@app.post("/select_sample")
def select_sample(sel: SampleSelect):
    """Switch active sample, reload arrays, and regenerate outputs."""
    global cached_full_trajectory

    try:
        cfg = _load_config()
        sample_dir = _resolve_sample_dir(cfg, sel.specie, sel.tree_ID, sel.disk_ID)

        # Clean previous artifacts to avoid stale data
        for p in [FULL_GEOM_JSON, TRAJECTORY_JSON]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        _load_sample_data(sample_dir)

        # Rebuild the compact JSON & trajectory
        data = {
            "sources": [],
            "detector_panels": [],
            "fov_rays": [],
            "full_trajectory": [],
            "bezier_curves": []
        }
        for i in range(N):
            src, corners, rays = get_geometry_at(i)
            bezier = get_bezier_surface_points(i)
            data["sources"].append(src)
            data["detector_panels"].append(corners)
            data["fov_rays"].append(rays)
            data["full_trajectory"].append(src)
            data["bezier_curves"].append(bezier)

        with open(FULL_GEOM_JSON, "w") as f:
            json.dump(data, f)

        cached_full_trajectory = data["full_trajectory"]
        with open(TRAJECTORY_JSON, "w") as f:
            json.dump(cached_full_trajectory, f)

        return {"status": "ok", "sample_dir": str(sample_dir), "frames": N}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


#########################################################
# get_sinogram_slice (route)
# Export a single projection as a 2D NRRD (with singleton Z)
# to make it easy for 3D Slicer to display/scroll slices.
# - Validates bounds and shape
# - Writes a temporary .nrrd and serves it
#########################################################
@app.get("/get_sinogram_slice/{index}")
def get_sinogram_slice(index: int):
    print("[DEBUG] Requested index:", index)
    print("[DEBUG] sinogram shape:", sinogram.shape)

    if sinogram.ndim != 3:
        return JSONResponse(
            status_code=500,
            content={"error": f"Expected 3D sinogram, got shape: {sinogram.shape}"}
        )

    max_index = sinogram.shape[0]
    if index < 0 or index >= max_index:
        return JSONResponse(
            status_code=400,
            content={"error": f"Index {index} out of bounds. Valid range: 0..{max_index - 1}"}
        )

    # Extract [det_x, det_z] then transpose to [det_z, det_x] for conventional display
    slice_2d = sinogram[index, :, :].T
    slice_3d = slice_2d[:, :, np.newaxis]  # [det_z, det_x, 1]

    # Build a simple NRRD header with isotropic detector pixel spacing
    pix_size = float(metadata.get("DET_PIX_SIZE", 1.0))
    header = {
        "type": "float",
        "dimension": 3,
        "sizes": list(slice_3d.shape),
        "space": "left-posterior-superior",
        "space directions": [
            [pix_size, 0.0, 0.0],                # X → det_z
            [0.0, pix_size, 0.0],                # Y → det_x
            [0.0, 0.0, 1.0],                     # Z singleton
        ],
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }

    # Write to a temporary file and return it
    temp_file = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    nrrd.write(temp_file.name, slice_3d.astype(np.float32), header)

    return FileResponse(temp_file.name, media_type="application/octet-stream", filename=f"sinogram_{index}.nrrd")


# ----- Run reconstruction inside this container (no SSH/Apptainer)
class ReconRequest(BaseModel):
    specie: str
    tree_ID: int
    disk_ID: int
    method: str                  # "fbp" | "landweber" | "adjoint" | ...
    parameters: dict | None = None


#########################################################
# get_supported_methods
# Try to parse reconstruction.py for the argparse choices
# of --reconstruction_method; otherwise fall back to a
# safe default set.
#########################################################
def get_supported_methods() -> list[str]:
    """
    Try to discover allowed methods from reconstruction.py by parsing the
    argparse 'choices=[...]' for --reconstruction_method. Falls back
    to a safe default if parsing fails.
    """
    try:
        rp = BASE_DIR / "reconstruction.py"
        txt = rp.read_text(encoding="utf-8", errors="ignore")
        # look for: choices=['adjoint','fbp','landweber'] or ["adjoint", "fbp", "landweber"]
        import re, ast
        m = re.search(r"choices\s*=\s*\[([^\]]+)\]", txt)
        if m:
            # Build a Python list literal: '[' + captured + ']'
            raw = "[" + m.group(1) + "]"
            vals = ast.literal_eval(raw)
            # Normalize and keep strings only
            methods = sorted({str(v).lower() for v in vals})
            if methods:
                return methods
    except Exception:
        pass

    # Safe fallback if anything goes wrong
    return ["adjoint", "fbp", "landweber"]


#########################################################
# run_reconstruction (route)
# Trigger a reconstruction by calling reconstruction.py
# with selected method and optional parameters. Returns
# the produced NRRD path (served under /images).
#
# Typical client flow:
# 1) POST /run_reconstruction with JSON {specie, tree_ID, disk_ID, method, parameters?}
# 2) Poll/handle response; download NRRD via "url" field.
#########################################################
@app.post("/run_reconstruction")
def run_reconstruction(req: ReconRequest, request: Request):
    supported_methods = get_supported_methods()
    method = req.method.lower()

    if method not in supported_methods:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported method '{req.method}'.",
                "allowed": supported_methods
            }
        )

    # Validate sample exists (re-uses config + path helpers)
    cfg = _load_config()
    sample_dir = _resolve_sample_dir(cfg, req.specie, req.tree_ID, req.disk_ID)
    if not sample_dir.exists():
        return JSONResponse(status_code=400, content={"error": f"Sample not found: {sample_dir}"})

    # Ensure output dir
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    out_name = f"tree{req.tree_ID}_disk{req.disk_ID}_{method}.nrrd"
    out_path = IMAGES_DIR / out_name

    # Build command for reconstruction.py
    cmd = [
        "python", "-u", "reconstruction.py",
        "--tree_ID", str(req.tree_ID),
        "--disk_ID", str(req.disk_ID),
        "--reconstruction_method", method,
        "--output_folder", str(IMAGES_DIR),
    ]
    # Pass optional parameters as a JSON blob to the script
    if req.parameters:
        cmd += ["--parameters", json.dumps(req.parameters)]

    print("[INFO] Starting reconstruction:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    print("[INFO] Reconstruction finished with code:", proc.returncode)

    if proc.returncode != 0:
        # bubble up some context to the client
        return JSONResponse(
            status_code=500,
            content={
                "error": "Reconstruction failed",
                "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-2000:]
            }
        )

    if not out_path.exists():
        return JSONResponse(
            status_code=500,
            content={"error": f"Expected output not found: {out_path}"}
        )

    # Minimal JSON with path to the produced image under /images
    return JSONResponse(
        content=json.loads(json.dumps({
            "status": "ok",
            "method": method,
            "allowed_methods": supported_methods,
            "filename": out_name,
            "url": f"/images/{out_name}",
            # "stdout": proc.stdout[-2000:],  # tail for quick inspection (opt-in)
        }, indent=2))
    )


#########################################################
# root (route)
# Small landing route with usage hints and supported methods.
#########################################################
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "CT streaming & reconstruction server",
        "supported_methods": get_supported_methods(),
        "use": {
            "POST /run_reconstruction": {
                "json": {
                    "specie": "pine",
                    "tree_ID": 16,
                    "disk_ID": 1,
                    "method": "|".join(get_supported_methods()),
                    "parameters": {"niter": 80, "omega": 0.5}
                }
            },
            "GET /images/<filename>": "download reconstructed NRRD",
            "GET /stream_window": "geometry window",
            "GET /full_geometry": "full geometry JSON"
        }
    }