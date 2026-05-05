from typing import Tuple, Dict
import odl
from odl.applications.tomo.geometry.conebeam import ConeBeamGeometry
import numpy as np
from functools import partial
from scipy.interpolate import interp1d

from odl.contrib.torch import OperatorModule

def parser_ConeBeamGeometry(
        angles:np.ndarray,
        axial_positions:np.ndarray,
        metadata:Dict,
        torch = False
    ):
    DET_X_MIN = metadata['DET_X_MIN']
    DET_X_MAX = metadata['DET_X_MAX']
    DET_NPX_X = metadata['DET_NPX_X']
    DET_Z_MIN = metadata['DET_Z_MIN']
    DET_Z_MAX = metadata['DET_Z_MAX']
    DET_NPX_Z = metadata['DET_NPX_Z']
    detector_partition = odl.uniform_partition(
            [DET_X_MIN, DET_Z_MIN],
            [DET_X_MAX, DET_Z_MAX],
            (DET_NPX_X, DET_NPX_Z))
    
    axial_positions -= axial_positions[0]
    #axial_positions += 230*0.3
    axial_positions += metadata["DET_NPX_Z"] * metadata["DET_PIX_SIZE"]

    REC_MIN_X, REC_MIN_Y, REC_MIN_Z = metadata['REC_MIN_X'], metadata['REC_MIN_Y'], axial_positions[0]
    REC_MAX_X, REC_MAX_Y, REC_MAX_Z = metadata['REC_MAX_X'], metadata['REC_MAX_Y'], axial_positions[-1]
    REC_NPX_X, REC_NPX_Y, REC_NPX_Z = metadata['REC_NPX_X'], metadata['REC_NPX_Y'], int((REC_MAX_Z - REC_MIN_Z) // metadata['REC_PIC_SIZE'])

    reco_space = odl.uniform_discr(
        min_pt=[REC_MIN_X, REC_MIN_Y, REC_MIN_Z],
        max_pt=[REC_MAX_X, REC_MAX_Y, REC_MAX_Z],
        shape= [REC_NPX_X, REC_NPX_Y, REC_NPX_Z],
        dtype='float32')

    SRC_RADIUS = metadata['SRC_RADIUS']
    DET_RADIUS = metadata['DET_RADIUS'] 
    DET_CURVATURE_RADIUS = metadata['DET_CURVATURE_RADIUS']

    angles_increasing = np.unwrap(angles)
    angle_partition = odl.nonuniform_partition(angles_increasing)
    
    # The whole compute_z_shifts function seems a little complicated.
    # Here is how this is handled in odl_stream_server.py which seems simpler.
    # Julius Häger - 2026-03-26
    # Keep an angle→z interpolation
    z_shift_func = interp1d(
        angles_increasing, axial_positions, kind="linear",
        bounds_error=False, fill_value=(axial_positions[0], axial_positions[-1]) # type: ignore The function has a special case for fill_value being a 2-tuple.
    )

    def shift_func(angle):
        # FIXME: Use
        #np.interp(angle, angles_increasing, z_corrected)
        res = np.zeros((len(angle), 3))
        res[:, 2] = z_shift_func(angle)
        return res

    geometry = ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=metadata["SRC_RADIUS"],
        det_radius=metadata["DET_RADIUS"],
        det_curvature_radius=(metadata["DET_CURVATURE_RADIUS"], None),
        pitch=0, # type: ignore The argument is a float, the function annotation is wrong.
        axis=[0, 0, 1],
        src_shift_func=shift_func,     # could be set to a function of angle if needed
        det_shift_func=shift_func,
        translation=[0, 0, 0],
    )
    ray_trafo = odl.applications.tomo.operators.ray_trafo.RayTransform(reco_space, geometry, impl='astra_cuda')
    ray_trafo_adjoint = ray_trafo.adjoint
    if torch:
        return OperatorModule(ray_trafo), OperatorModule(ray_trafo_adjoint)
    else:
        return ray_trafo, ray_trafo_adjoint