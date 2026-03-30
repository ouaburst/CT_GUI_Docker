from typing import Tuple, Dict
import odl
import numpy as np
from functools import partial

from odl.contrib.torch import OperatorModule

def compute_z_shifts(
        angles:np.ndarray,
        axial_positions:np.ndarray,
        ) -> Tuple[np.ndarray, int]:
        z_shifts = angles.copy()
        section_indices = [0] + list(np.where(np.diff(angles) < 0)[0] +1)  + [len(z_shifts)+1]
        pitches = []
        theta_ac = 0
        delta_ac = 0 
        for index in range(len(section_indices)-1):       
            section_start = section_indices[index]
            section_end   = section_indices[index+1]
            section_positions = axial_positions[section_start:section_end] - axial_positions[section_start]
            section_angles    = angles[section_start:section_end] - angles[section_start]
            delta_z     = section_positions[-1]-section_positions[0]
            delta_theta = section_angles[-1]-section_angles[0]
            delta_ac += delta_z
            theta_ac += delta_theta
            # For each section, we compute the expected pitch should it be uniform
            pitch_value = delta_z / delta_theta
            pitches.append(pitch_value*2*np.pi)        
            normalised_pitch:np.ndarray = section_angles  * pitch_value
            z_shifts[section_start:section_end] = section_positions - normalised_pitch 

        pitch = int(delta_ac / (theta_ac / (2*np.pi)))
        z_shifts_array = np.transpose(np.vstack(
            [
                np.zeros(len(angles)), 
                np.zeros(len(angles)), 
                z_shifts
            ]))
        return z_shifts_array, pitch

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
    axial_positions += 230*0.3

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
    
    ### Compute shifts
    shifts, PITCH = compute_z_shifts(angles, axial_positions)

    angle_partition = odl.nonuniform_partition(np.unwrap(angles))
    # Source Shift
    src_shift_func = partial(
        odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts
        )
    # Detector shift 
    det_shift_func = partial(
        odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts
        )

    # NB: det_radius is not equal det_curvature_radius!
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SRC_RADIUS,
        det_radius=DET_RADIUS,
        det_curvature_radius=(DET_CURVATURE_RADIUS,None),  # uncomment for curved detector
        pitch=PITCH,
        src_shift_func=src_shift_func,
        det_shift_func=det_shift_func)
    
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
    ray_trafo_adjoint = ray_trafo.adjoint
    if torch:
        return OperatorModule(ray_trafo), OperatorModule(ray_trafo_adjoint)
    else:
        return ray_trafo, ray_trafo_adjoint