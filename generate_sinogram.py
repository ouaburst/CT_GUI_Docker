"""Example using the ray transform with circular cone beam geometry."""

import numpy as np
import odl
import typing
import json
from pathlib import Path

detector_resolution = [512, 512]

# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 300],
    dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-np.pi/32, -20], [np.pi/32, 20], [512, 512])
geometry = odl.applications.tomo.ConeBeamGeometry(
    angle_partition, detector_partition, src_radius=200,
    det_radius=200, det_curvature_radius=(400, None),
    # FIXME: Generating with a different axis doesn't work...
    axis=[0, 0, 1], pitch=20)

# Ray transform (= forward projection).
ray_trafo = odl.applications.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.core.phantom.shepp_logan(reco_space, True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Write out the npy files.
DATA_FOLDER = Path("/test/data/real_datasets/ml_ready/")
SAMPLE_NAME = "phantom_1_1" # FIXME: Match the sample index structure...
SAMPLE_FOLDER = DATA_FOLDER / SAMPLE_NAME

metadata = {
    "GEOMETRY_ENGINE": "ODL",
    "GEOMETRY_NAME": "ConeBeamGeometry",
    "DET_NPX_Z": detector_partition.shape[0],
    "DET_NPX_X": detector_partition.shape[1],
    "DET_PIX_SIZE": detector_partition.cell_sides[1], # FIXME: aniso pixel size?
    "DET_X_MIN": detector_partition.min_pt[0],
    "DET_X_MAX": detector_partition.max_pt[0],
    "DET_Z_MIN": detector_partition.min_pt[1],
    "DET_Z_MAX": detector_partition.max_pt[1],
    "SRC_RADIUS": geometry.src_radius,
    "DET_RADIUS": geometry.det_radius,
    "DET_CURVATURE_RADIUS": typing.cast(odl.applications.tomo.CylindricalDetector, geometry.detector).radius,
    "REC_PIC_SIZE": reco_space.cell_sides[0], # FIXME: Aniso-pixels?
    "REC_NPX_X": reco_space.shape[0],
    "REC_NPX_Y": reco_space.shape[1],
    "REC_MIN_X": reco_space.min_pt[0],
    "REC_MAX_X": reco_space.max_pt[0],
    "REC_MIN_Y": reco_space.min_pt[1],
    "REC_MAX_Y": reco_space.max_pt[1],
    # FIXME: If the detector doesn't have a pitch the default ROI is completely off as we use that for the ROI Z extent.
}

import os
import stat

SAMPLE_FOLDER.mkdir(mode=0o777, parents=True)
with open(SAMPLE_FOLDER / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
os.chmod(SAMPLE_FOLDER / "metadata.json", stat.S_IWOTH | stat.S_IROTH | stat.S_IWGRP | stat.S_IRGRP | stat.S_IWUSR | stat.S_IRUSR)
print(metadata)

np.save(SAMPLE_FOLDER / "sinogram.npy", proj_data.data, False)
np.save(SAMPLE_FOLDER / "angles.npy", geometry.angles, False)
# FIXME: Get the correct shape and values
np.save(SAMPLE_FOLDER / "axial_positions.npy", geometry.src_position(geometry.angles)[:, 2])
np.save(SAMPLE_FOLDER / "shifts", np.zeros((geometry.angles.shape[0], 3)))

# FIXME: Describe the samples in a separate file from the reconstruction methods...
server_config = {
    "samples": [ { "specie": "phantom", "tree_ID": 1, "disk_ID": 1  } ]
}

print(json.dumps(server_config))