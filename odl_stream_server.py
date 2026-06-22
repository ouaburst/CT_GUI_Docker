from fastapi import FastAPI, Query, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import logging
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import json
import odl
from odl.applications.tomo.geometry.conebeam import ConeBeamGeometry
from scipy.interpolate import interp1d
import tempfile
import nrrd
import subprocess
import argparse
import runpy
import gzip
import time
import PIL
import PIL.Image
import multiprocessing
import typing
from filelock import FileLock, Timeout

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
sample_config: dict
sinogram: np.typing.NDArray[np.float32] # numpy array with shape (N, det_x, det_z)
sinogramMin: np.float32 # Minimum value in the sinogram data
sinogramMax: np.float32 # Maximum value in the sinogram data
angles = None            # raw angles array (not directly used after sort)
z_positions = None       # raw axial positions (not directly used after correction)
shifts = None            # per-projection shifts (x,y,z)
metadata: dict           # dict with geometry + reconstruction grid settings
geometry: ConeBeamGeometry # ODL ConeBeamGeometry constructed from dataset
N = 0                    # number of projections
pix_size_det = 0.0       # detector pixel size (mm)
cached_geometry: dict   # the geometry data
sample_name: str



# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Directory where reconstruction outputs (NRRDs) will be saved/served
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

RECONSTRUCTION_CONFIG_PATH = BASE_DIR / "reconstruction_methods.json"
SAMPLE_CONFIG_PATH = BASE_DIR / "sample_config.json"
FULL_GEOM_NPZ = OUTPUT_DIR / "full_geometry.npz"

SINOGRAM_CACHE_DIR = OUTPUT_DIR / "sinogram_cache"


#########################################################
# _load_sample_config
# Load sample config JSON which describes available samples
# and the bind-mounted volume root.
#########################################################
def _load_sample_config() -> dict:
    if not SAMPLE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {SAMPLE_CONFIG_PATH}")
    with open(SAMPLE_CONFIG_PATH) as f:
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
    # FIXME: real_datasets/ml_ready should be part of the config...
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
    global sinogram, sinogramMin, sinogramMax, angles, z_positions, shifts, metadata, geometry, N, pix_size_det

    print(f"[INFO] Loading data from: {sample_dir}")
    req = ["sinogram.npy", "angles.npy", "axial_positions.npy", "shifts.npy", "metadata.json"]
    missing = [p for p in req if not (sample_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {sample_dir}: {missing}")

    # Load raw arrays
    start = time.time()
    #sinogram = np.load(sample_dir / "sinogram.npy") # expected (N, det_x, det_z)
    sinogram = np.load(sample_dir / "sinogram.npy", mmap_mode="r") # expected (N, det_x, det_z)
    angles_raw = np.load(sample_dir / "angles.npy")
    z_raw = np.load(sample_dir / "axial_positions.npy")
    shifts_raw = np.load(sample_dir / "shifts.npy")
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)
    # FIXME: This takes a long time, if we had these precomputed switching samples would be pretty fast.
    # - Julius Häger 2026-03-31
    sinogramMin = np.min(sinogram)
    sinogramMax = np.max(sinogram)
    end = time.time()
    print(f"[INFO] Sinogram min {sinogramMin} max {sinogramMax}")
    print(f"[INFO] Sinogram loaded with shape {sinogram.shape} and dtype {sinogram.dtype} in {end-start:.3} s")

    start = time.time()
    # Apply z correction (accounting for any per-view shift in z)
    z_corrected = z_raw + shifts_raw[:, 2]

    # Cache counts and detector pixel size
    N = len(angles_raw)
    pix_size_det = float(metadata["DET_PIX_SIZE"])

    # The angles are all mod 2PI so we need to undo this operation
    # - Julius Häger 2026-03-26
    angles_increasing = np.unwrap(angles_raw)

    # Estimate helical pitch (z advance per full 2π revolution)
    angle_range = float(angles_increasing[-1] - angles_increasing[0])
    z_range = float(z_corrected[-1] - z_corrected[0])
    pitch = z_range * (2 * np.pi) / angle_range if angle_range != 0 else 0.0
    print(f"[INFO] Computed average pitch: {pitch:.2f} mm")

    angle_partition = odl.nonuniform_partition(angles_increasing)

    # Create 2D detector partition using physical extents and number of pixels
    detector_partition = odl.uniform_partition(
        [metadata["DET_X_MIN"], metadata["DET_Z_MIN"]],
        [metadata["DET_X_MAX"], metadata["DET_Z_MAX"]],
        [metadata["DET_NPX_X"], metadata["DET_NPX_Z"]],
    )

    # Keep an angle→z interpolation
    z_shift_func = interp1d(
        angles_increasing, z_corrected, kind="linear",
        bounds_error=False, fill_value=(z_corrected[0], z_corrected[-1]) # type: ignore The function has a special case for fill_value being a 2-tuple.
    )

    def shift_func(angle):
        res = np.zeros((len(angle), 3))
        res[:, 2] = z_shift_func(angle)
        return res

    # Build helical cone-beam geometry; det_curvature_radius[0] is tangential curvature
    geometry = ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=metadata["SRC_RADIUS"],
        det_radius=metadata["DET_RADIUS"],
        det_curvature_radius=(metadata["DET_CURVATURE_RADIUS"], None),
        pitch=0, # type: ignore The argument is a float, the function annotation is wrong.
        axis=metadata["ROTATION_AXIS"],
        src_shift_func=shift_func,     # could be set to a function of angle if needed
        det_shift_func=shift_func,
        translation=[0, 0, 0],
    )

    # Optional sanity checks against reconstruction grid definition
    _validate_reconstruction_geometry(metadata)

    end = time.time()
    print(f"Creating odl geometry took {end-start:.3} seconds")

#########################################################
# generate_sensor_geometry
# Generate geometry for all indices.
# Approximate the curved detector panel as a quadratic Bézier
# surface mesh for visualization (num_v × num_u control).
# 4 FOV rays and the source position for a given index i.
#########################################################
def generate_sensor_geometry(geometry: ConeBeamGeometry, num_u=8, num_v=4):
    angles = geometry.angles
    src_positions = geometry.src_position(angles)
    det_refpoints = geometry.det_refpoint(angles)
    det_axes = geometry.det_axes(angles)

    du = det_axes[:, 0, :]
    dv = det_axes[:, 1, :]

    #print(f"src_positions = {src_positions.shape}")
    #print(f"det_refpoints = {det_refpoints.shape}")
    #print(f"det_axes = {det_axes.shape}")
    #print(f"du = {du.shape}")
    #print(f"dv = {dv.shape}")

    # Determine tangential curvature range from pixel count and pixel size
    R_curv = float(metadata["DET_CURVATURE_RADIUS"])
    n_x = int(metadata["DET_NPX_X"])
    arc_length = n_x * pix_size_det
    theta_range = arc_length / R_curv

    # Use 3 control columns in theta (quadratic Bézier) and 3 rows in z
    theta_vals = np.linspace(-theta_range / 2, theta_range / 2, 3)
    z_vals = np.linspace(float(metadata["DET_Z_MIN"]), float(metadata["DET_Z_MAX"]), 3)

    # Outward normal approx from detector center to source
    normals = src_positions - det_refpoints
    normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    # Build 3×3 control grid in 3D
    # FIXME: Change the order of dimensions in some way that makes sense...
    control_points = np.empty((3, 3, len(angles), 3))
    for iz, z in enumerate(z_vals):
        for itheta, theta in enumerate(theta_vals):
            # Arc along du + sagitta along normal + vertical offset along dv
            control_points[iz, itheta] = det_refpoints + R_curv * np.sin(theta) * du + R_curv * (1 - np.cos(theta)) * normals + z * dv

    #print(f"control_points: {control_points.shape}")

    # Local helper: 1D quadratic Bézier interpolation for 3 control pts
    def bezier_interp_1d(ctrl, t: float):
        return (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * ctrl[2]

    # Sample u (tangential) and v (vertical)
    u_vals = np.linspace(0, 1, num_u)
    v_vals = np.linspace(0, 1, num_v)

    v: np.float64
    u: np.float64
    surface_points = np.empty((num_v, num_u, len(angles), 3))
    surface_uvs = np.empty((num_v, num_u, 2))
    interm_rows = np.empty((3, len(angles), 3))
    for iv, v in enumerate(v_vals):
        for iu, u in enumerate(u_vals):
            interm_rows[0] = bezier_interp_1d(control_points[0], u)
            interm_rows[1] = bezier_interp_1d(control_points[1], u)
            interm_rows[2] = bezier_interp_1d(control_points[2], u)
            pt = bezier_interp_1d(interm_rows, v)
            surface_points[iv, iu] = pt
            surface_uvs[iv, iu] = np.array([u, v])

    surface_points = np.transpose(surface_points, (2, 0, 1, 3)) # (N, num_v, num_u, 3)
    #print(f"surface_points: {surface_points.shape}")
    #print(f"surface_uvs: {surface_uvs.shape}")

    rays = np.empty((4, 2, len(src_positions), 3))

    rays[0, 0] = src_positions
    rays[1, 0] = src_positions
    rays[2, 0] = src_positions
    rays[3, 0] = src_positions

    rays[0, 1] = surface_points[:,  0,  0, :]
    rays[1, 1] = surface_points[:,  0, -1, :]
    rays[2, 1] = surface_points[:, -1, -1, :]
    rays[3, 1] = surface_points[:, -1,  0, :]

    rays = np.transpose(rays, (2, 0, 1, 3)) # (N, 4, 2, 3)

    return src_positions, rays, surface_points, surface_uvs # (N, 3) (N, 4, 2, 3) (N, num_v, num_u, 3), (num_v, num_u, 2)

# ----------------------------------
# FastAPI app
# ----------------------------------
app = FastAPI()

logger = logging.getLogger(__name__)

# Serve static reconstruction outputs (NRRDs produced by run_reconstruction)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

@app.middleware("http")
async def log_to_access_time(request: Request, call_next):
    start = time.time()
    try:
        return await call_next(request)
    finally:
        end = time.time()
        logger.warning(f"Info:\t{request.method} {request.url} took {end-start:.3} s")

#########################################################
# startup (startup)
# On app startup:
# - Load config and the first sample
# - Precompute and persist a compact geometry JSON
# - Persist the full trajectory JSON
# - Generate sinogram image cache
#########################################################
@app.on_event("startup")
def startup():
    """Load config + default sample, then write output/full_geometry.json and cache trajectory."""
    global sample_config, sample_name

    start = time.time()
    sample_config = _load_sample_config()
    if not sample_config.get("samples"):
        raise RuntimeError("No samples listed in sample_config.json")
    first = sample_config["samples"][0]
    sample_name = ""
    select_sample(SampleSelect(specie=first["specie"], tree_ID=int(first["tree_ID"]), disk_ID=int(first["disk_ID"])))
    end = time.time()
    print(f"[INFO] Startup took {end - start} seconds")

#########################################################
# get_reconstruction_methods (route)
# Serve reconstruction_config.json, a list of supported reconstruction methods and their parameters.
#########################################################
@app.get("/reconstruction_methods.json")
def get_reconstruction_methods():
    return FileResponse(RECONSTRUCTION_CONFIG_PATH, media_type="application/json", filename="reconstruction_methods.json")

#########################################################
# get_sample_config (route)
# Serve sample_config.json, a list of samples.
#########################################################
@app.get("/sample_config.json")
def get_sample_config():
    return FileResponse(SAMPLE_CONFIG_PATH, media_type="application/json", filename="sample_config.json")

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
# full_dataset (route)
# Send the entire sinogram and per-index geometry payload.
# WARNING: Can be large; intended for debugging/exports,
# not real-time streaming to clients.
#########################################################
@app.get("/full_dataset")
def full_dataset():
    global cached_geometry

    print("[INFO] Dataset ready for transmission.")
    return JSONResponse(content={
        "total_angles": N,
        "sinogram": sinogram.tolist(),  # shape: (N, det_x, det_z)
        "geometry": cached_geometry,    # geometry data
    })

#########################################################
# get_full_geometry_npz (route)
# Serve the compact precomputed full geometry NPZ file.
#########################################################
@app.get("/full_geometry_npz")
def get_full_geometry_npz():
    if not FULL_GEOM_NPZ.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(FULL_GEOM_NPZ, media_type="application/x-npz")

#########################################################
# serve_full_trajectory (route)
# Serve only the full source trajectory JSON.
#########################################################
@app.get("/full_trajectory.json")
def serve_full_trajectory():
    global cached_geometry
    return JSONResponse(content=cached_geometry["full_trajectory"])

# ----- Switch sample at runtime (optional)
class SampleSelect(BaseModel):
    specie: str
    tree_ID: int
    disk_ID: int

#########################################################
# select_sample (route)
# Switch active dataset (specie/tree/disk), reload arrays,
# and rebuild cached geometry + trajectory JSONs.
# This request takes around 30 seconds to complete if
# the sinogram cache needs to be created.
#########################################################
@app.post("/select_sample")
def select_sample(sel: SampleSelect):
    """Switch active sample, reload arrays, and regenerate outputs."""
    global cached_geometry, sample_name, sample_config, sinogram, sinogramMin, sinogramMax, metadata, sinogram

    try:
        sample_dir = _resolve_sample_dir(sample_config, sel.specie, sel.tree_ID, sel.disk_ID)
        new_sample_name = f"{sel.specie}_{sel.tree_ID}_{sel.disk_ID}"
        if sample_name == None or sample_name != new_sample_name:
            sample_name = new_sample_name
            print(f"[INFO] Selecting sample {sample_name}...")
            _load_sample_data(sample_dir)
            
            print("[INFO] Precomputing and saving full geometry JSON to disk...")

            # Rebuild the compact JSON & trajectory
            cached_geometry = {
                "sources": [],
                "detector_panels": [],
                "fov_rays": [],
                "full_trajectory": [],
                "bezier_curves": [],
                "bezier_curves_uvs": []
            }

            start = time.time()
            src, rays, surface, surface_uv = generate_sensor_geometry(geometry)
            cached_geometry["sources"] = src.tolist()
            cached_geometry["bezier_curves"] = surface.tolist()
            cached_geometry["bezier_curves_uvs"] = surface_uv.tolist()
            cached_geometry["fov_rays"] = rays.tolist()
            cached_geometry["full_trajectory"] = src.tolist()
            end = time.time()
            print(f"[DEBUG] Generating sensor geometry took {end - start} seconds")

            start = time.time()
            #with open(FULL_GEOM_JSON, "wt") as f:
            #    json.dump(cached_geometry, f)

            np.savez_compressed(FULL_GEOM_NPZ,
                                allow_pickle=False,
                                sources=src.astype(np.float32),
                                bezier_curves=surface.astype(np.float32),
                                bezier_curves_uvs=surface_uv.astype(np.float32),
                                fov_rays=rays.astype(np.float32),
                                full_trajectory=src.astype(np.float32))

            #with gzip.open(FULL_GEOM_JSON_GZIP, "wt", encoding="utf-8") as f:
            #    json.dump(cached_geometry, f)
            end = time.time()
            print(f"[DEBUG] Saving geometry and trajectory took {end - start} seconds")

            print(f"[INFO] Full geometry saved to {FULL_GEOM_NPZ}")

            sample_sinogram_cache_dir = SINOGRAM_CACHE_DIR / sample_name
            Path.mkdir(sample_sinogram_cache_dir, parents=True, exist_ok=True)

            def generate_sinogram_cache():
                start = time.time()
                print(f"[INFO] Generating sinogram cache (jp2)")
                sample_sinogram_cache_dir = SINOGRAM_CACHE_DIR / sample_name
                # FIXME: Use cores - 1 instead of 32?
                with multiprocessing.Pool(processes=32) as p:
                    p.map(compress_sinogram_slice_jp2, range(N))
                    # FIXME: Some way to report progress...?
                    p.close()
                    p.join()
                end = time.time()
                
                import os
                jp2_size = 0
                for i in range(N):
                    jp2_size += os.path.getsize(sample_sinogram_cache_dir / f"{i}.jp2")
                print(f"[INFO] Generated sinogram cache in {end - start} seconds ({jp2_size} b)")

            multiprocessing.Process(target=generate_sinogram_cache).start()
        else:
            print(f"[DEBUG] Sample '{new_sample_name}' was already selected, doing nothing.")

        return {"status": "ok", "sample_dir": str(sample_dir), "frames": N, "sinogram_shape": sinogram.shape, "sinogram_min": float(sinogramMin), "sinogram_max": float(sinogramMax), "metadata": metadata}
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def compress_sinogram_slice_jp2(i):
    global sample_name, sinogram, sinogramMin, sinogramMax
    sample_sinogram_cache_dir = SINOGRAM_CACHE_DIR / sample_name
    
    file_path = sample_sinogram_cache_dir / f"{i}.jp2"
    json_file_path = sample_sinogram_cache_dir / f"{i}.jp2.json"
    lock_path = sample_sinogram_cache_dir / f"{i}.jp2.lock"
    lock = FileLock(lock_path)
    with lock.acquire(timeout=3):
        if Path.exists(file_path) == False:
            slice_2d = sinogram[i, :, :].T
            slice_min = np.min(slice_2d)
            slice_max = np.max(slice_2d)
            img_data = np.iinfo(np.uint8).max * (slice_2d - slice_min) / (slice_max - slice_min)
            im = PIL.Image.fromarray(img_data.astype(np.uint8))
            im.save(file_path, format="", irreversible=True, quality_mode="dB", quality_layers=[44])
            # Save the dynamic range of the jp2 file so we can recover the correct values in the client
            with open(json_file_path, 'w') as f:
                json.dump({ "slice_min": repr(float(slice_min)), "slice_max": repr(float(slice_max)) }, f)



def compress_sinogram_slice_avif(i):
    global sample_name, sinogram, sinogramMin, sinogramMax
    sample_sinogram_cache_dir = SINOGRAM_CACHE_DIR / sample_name

    file_path = sample_sinogram_cache_dir / f"{i}.avif"
    json_file_path = sample_sinogram_cache_dir / f"{i}.avif.json"
    lock_path = sample_sinogram_cache_dir / f"{i}.avif.lock"
    lock = FileLock(lock_path)
    with lock.acquire(timeout=3):
        if Path.exists(file_path) == False:
            slice_2d = sinogram[i, :, :].T
            slice_min = np.min(slice_2d)
            slice_max = np.max(slice_2d)
            img_data = np.iinfo(np.uint8).max * (slice_2d - slice_min) / (slice_max - slice_min)
            im = PIL.Image.fromarray(img_data.astype(np.uint8))
            im.save(file_path, format="", max_threads=1, quality=57, speed=7, subsampling="4:0:0")
            # Save the dynamic range of the avif file so we can recover the correct values in the client
            with open(json_file_path, 'w') as f:
                json.dump({ "slice_min": repr(float(slice_min)), "slice_max": repr(float(slice_max)) }, f)

#########################################################
# get_sinogram_slice (route)
# Export a single projection as a 2D NRRD (with singleton Z)
# to make it easy for 3D Slicer to display/scroll slices.
# - Validates bounds and shape
# - Writes a temporary .nrrd and serves it
#########################################################
@app.get("/get_sinogram_slice/{index}")
def get_sinogram_slice(index: int):
    print(f"[DEBUG] Requested index: {index}")
    print(f"[DEBUG] sinogram shape: {sinogram.shape}")

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
    start = time.time()
    temp_file = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    nrrd.write(temp_file.name, slice_3d.astype(np.float32, copy=False), header)
    end = time.time()
    print(f"[DEBUG] Writing nrrd file took: {end - start} secodns")

    return FileResponse(temp_file.name, media_type="application/octet-stream", filename=f"sinogram_{index}.nrrd")

#########################################################
# get_sinogram_slice_fast (route)
# Export a single projection as a JPEG 2000 image
#########################################################
@app.get("/get_sinogram_slice_fast/{index}")
def get_sinogram_slice_fast(index: int):
    global sample_name
    print(f"[DEBUG] Requested index: {index}")
    sample_sinogram_cache_dir = SINOGRAM_CACHE_DIR / sample_name
    print(f"[DEBUG] Cache dir: {sample_sinogram_cache_dir}")

    file = f"{index}.jp2"
    file_path = sample_sinogram_cache_dir / file
    meta_path = sample_sinogram_cache_dir / f"{index}.jp2.json"

    if Path.exists(file_path) and Path.exists(meta_path):
        print("[DEBUG] Sinogram cache hit")
    else:
        print("[DEBUG] Sinogram cache miss")
        try:
            compress_sinogram_slice_jp2(index)
        except Timeout as e:
            return Response(headers={'reason': 'timeout', 'message': f'Timeout while waiting for lock file \'{e.lock_file}\' for a long time. Stale lock or sinogram slice compression taking a long time?'}, status_code=500)
    
    with open(meta_path) as f:
        headers = json.load(f)
    return FileResponse(file_path, media_type="image/jp2", filename=file, headers=headers)

    #file = f"{index}.avif"
    #meta_file = f"{index}.avif.json"
    #with open(sample_sinogram_cache_dir / meta_file) as f:
    #    headers = json.load(f)
    #return FileResponse(sample_sinogram_cache_dir / file, media_type="image/avif", filename=file)

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

    print(f"[INFO] Got reconstruction request: {req}")

    if method not in supported_methods:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported method '{req.method}'.",
                "allowed": supported_methods
            }
        )

    # Validate sample exists (re-uses config + path helpers)
    cfg = _load_sample_config()
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
        "--specie", str(req.specie),
        "--tree_ID", str(req.tree_ID),
        "--disk_ID", str(req.disk_ID),
        "--reconstruction_method", method,
        "--output_folder", str(IMAGES_DIR),
    ]

    # Pass optional parameters as a JSON blob to the script
    if req.parameters:
        cmd += ["--parameters", json.dumps(req.parameters)]

    print(f"[INFO] Starting reconstruction: {' '.join(cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    end = time.time()
    print(f"[INFO] Reconstruction finished with code: {proc.returncode} in {end - start} seconds")

    print(proc.stdout)

    if proc.returncode != 0:
        print(f"[ERROR] Reconstruction: {proc.stderr}")
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
        print(f"[ERROR] Expected output not found: {out_path}")
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