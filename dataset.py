from typing import Dict
from pathlib import Path
from functools import partial
import math 

import pandas as pd
import numpy as np
import json
import odl
from odl.contrib.torch import OperatorModule

from torch.utils.data import Dataset

# Constant for detector width (scaled from 230 pixels to mm with factor 0.3)
DETECTOR_WIDTH = 230 * 0.3

#########################################################
# Parse ODL Geometry
# Parses projection geometry using angles, axial positions, 
# and metadata. Currently only supports ConeBeamGeometry.
#########################################################
def parse_ODL_geometry(
        angles :np.ndarray, 
        axial_positions:np.ndarray, 
        metadata:Dict, 
        torch:bool
        ):
    if metadata['GEOMETRY_NAME'] == 'ConeBeamGeometry':
        from odl_utils import parser_ConeBeamGeometry as geometry_parser
    else:
        raise NotImplementedError
    return geometry_parser(angles, axial_positions, metadata, torch)
        
#########################################################
# MITO Dataset
# Custom PyTorch Dataset for handling ML-ready CT data 
# from the MITO dataset. Supports training, validation,
# and testing splits, with optional loading of sinograms.
#########################################################
class MITO(Dataset):
    def __init__(
        self,
        mode = 'training',
        data_folder_path = Path('/media/Store-SSD/real_datasets/ml_ready'),
        training_proportion = 0.7,
        validation_proportion = 0.15,
        testing_proportion = 0.15, 
        load_sinogram = True
        ):
        # Ensure provided path exists
        assert data_folder_path.is_dir()
        self.data_folder_path = data_folder_path
        
        # Ensure mode is valid
        assert mode in ['training', 'validation', 'testing']

        # Ensure proportions sum to 1
        assert training_proportion + validation_proportion + testing_proportion == 1

        # Ensure load_sinogram is boolean
        assert isinstance(load_sinogram, bool)
        self.load_sinogram = load_sinogram

        # Path to CSV with dataset records
        data_records_path = data_folder_path.joinpath('ml_ready_data_records.csv')
        assert data_records_path.is_file()
        data_records = pd.read_csv(data_records_path)

        # Split dataset into training, validation, testing based on tree_ID
        unique_trees_indices = data_records['tree_ID'].unique()
        n_trees = len(unique_trees_indices)
        if mode == 'training':
            first_index = 0
            last_index  = math.floor(training_proportion * n_trees)
        elif mode == 'validation':
            first_index = math.floor(training_proportion * n_trees)
            last_index  = math.ceil((training_proportion + validation_proportion) * n_trees)
        else:  # testing
            first_index = math.ceil((training_proportion + validation_proportion) * n_trees)
            last_index  = None
        self.dataframe = data_records[data_records['tree_ID'].isin(
            unique_trees_indices[first_index:last_index]
            )]
    
    #########################################################
    # Length of Dataset
    # Returns the number of samples in the dataset.
    #########################################################
    def __len__(self):
        return self.dataframe.__len__()
        
    #########################################################
    # Get Item
    # Loads a single dataset sample:
    # - Loads metadata, angles, axial positions
    # - Builds forward/adjoint operators (A, A_T)
    # - Optionally loads the sinogram
    #
    # HOW THESE ARE USED LATER (typical workflows):
    # -------------------------------------------------------
    # * Forward projection (simulate measurements):
    #     y_sim = A(x) 
    #   where x is a 3D volume (ODL element shaped like the reconstruction space),
    #   and y_sim is a sinogram (ODL element in the data space).
    #
    # * Adjoint (backprojection):
    #     x_bp = A_T(y)
    #   where y is a sinogram (measured or simulated).
    #   Note: A_T is not the inverse; it's the adjoint/back-projector.
    #
    # * Filtered Backprojection (FBP):
    #   Often implemented as: x_fbp = FBP(y) = A_T( H * y )
    #   where H is a frequency-domain ramp-like filter (ODL has FBP operators).
    #
    # * Iterative reconstruction (e.g., Landweber / CG / PDHG):
    #     iterate: x_{k+1} = x_k - τ * A_T( A(x_k) - y_meas )
    #   possibly with regularization (e.g., TV), proximal steps, constraints.
    #
    # * Deep learning integration:
    #   Wrap A into a Torch module to include physics layers in a network:
    #       A_torch = OperatorModule(A)
    #       y_hat = A_torch(x_torch)
    #   Similarly, you can wrap A_T for learned iterative schemes / unrolled nets.
    #
    # SHAPES & PRACTICAL NOTES:
    # - ODL operators act on ODL space elements (not raw numpy arrays). 
    #   Conversion: space.element(np_array) and np.asarray(odl_elem).
    # - If your geometry/ODL backend is CUDA-enabled, the heavy ops run on GPU.
    # - For batching: ODL operators are not batched by default; apply per-sample
    #   or implement external batching logic.
    #########################################################
    def __getitem__(self, index) -> Dict:
        slice_row = self.dataframe.iloc[index]
        specie  = slice_row['specie']
        tree_ID = slice_row['tree_ID']
        disk_ID = slice_row['disk_ID']
        
        # Construct sample path
        sample_path = self.data_folder_path.joinpath(f'{specie}_{tree_ID}_{disk_ID}')
        
        # Load geometry information
        angles = np.load(sample_path.joinpath('angles.npy'))
        metadata = dict(json.load(open(sample_path.joinpath('metadata.json'))))
        axial_positions = np.load(sample_path.joinpath('axial_positions.npy'))
       
        # Create operators
        A, A_T = self.make_operators(angles, axial_positions, metadata)

        # Package outputs
        data_dict = {
            'A'   : A,     # Forward projector (volume -> sinogram)
            'A_T' : A_T,   # Adjoint/back-projector (sinogram -> volume)
        }

        # Optionally load sinogram (measured data). This is what you'd feed into
        # FBP or iterative methods as 'y_meas'. Keep in mind that when passing
        # to ODL you typically wrap this numpy array into the data space:
        #   y_elm = A.range.element(np_array) 
        if self.load_sinogram:
            data_dict['sinogram'] = np.load(sample_path.joinpath('sinogram.npy'))

        # Example (commented) usage sketch outside this method:
        # ---------------------------------------------------
        # y_meas_np = data_dict['sinogram']                         # numpy array
        # y_meas = A.range.element(y_meas_np)                       # ODL element
        # x_bp  = A_T(y_meas)                                       # backprojection
        # # FBP (if you created an FBP operator fbp_op separately):
        # # x_fbp = fbp_op(y_meas)
        #
        # # Torch integration (unrolled networks / learned priors):
        # # A_torch   = OperatorModule(A)
        # # AT_torch  = OperatorModule(A_T)
        # # x_torch, y_torch are torch.Tensors (properly shaped/typed)
        # # y_hat = A_torch(x_torch)
        # # x_bp_from_torch = AT_torch(y_torch)
        return data_dict
    
    #########################################################
    # Random Indices
    # Returns a permutation of indices for random sampling.
    #########################################################
    def random_indices(self):
        return np.random.permutation(self.__len__())
    
    #########################################################
    # Make Operators
    # Creates forward and adjoint projection operators using ODL
    # based on angles, axial positions, and metadata.
    #
    # WHAT THIS RETURNS:
    #   forward_operator = A   : maps reconstruction space -> data space
    #   adjoint_operator = A_T : maps data space -> reconstruction space
    #
    # TYPICAL DOWNSTREAM USE:
    # - Forward model in iterative schemes: A(x)
    # - Gradient step through data fidelity: A_T(A(x) - y)
    # - Physics layer inside DL models via OperatorModule
    #
    # PERFORMANCE NOTES:
    # - If metadata selects an ASTRA CUDA backend in your geometry parser,
    #   these ops will execute on GPU. ODL manages the heavy lifting.
    # - Memory layout matters: prefer float32 where possible, avoid 
    #   unnecessary copies between numpy/ODL/torch.
    #########################################################
    def make_operators(
        self,
        angles:np.ndarray,
        axial_positions:np.ndarray,
        metadata:Dict,
        local_metadata = None,
        torch = False
        ):
        # Merge with any local metadata overrides
        if local_metadata:
            for key, value in local_metadata.items():
                metadata[key] = value

        # Currently only ODL backend is supported
        if metadata['GEOMETRY_ENGINE'] == 'ODL':
            forward_operator, adjoint_operator = parse_ODL_geometry(
                angles, axial_positions, metadata, torch
            )
        else: 
            raise NotImplementedError
        
        # Sanity tip:
        # - forward_operator.domain == adjoint_operator.range (volume space)
        # - forward_operator.range  == adjoint_operator.domain (sinogram space)
        # You can inspect with: print(A.domain), print(A.range)
        return forward_operator, adjoint_operator

#########################################################
# Main Execution
# Example usage:
# - Initialize MITO dataset
# - Fetch one sample
# - Extract geometry and plot source positions in 3D
#########################################################
if __name__ == '__main__':
    dataset = MITO(mode='training', load_sinogram=False)
    sample = dataset.__getitem__(0)
    
    # Extract geometry from first sample
    geometry: odl.tomo.ConeBeamGeometry = sample['A'].geometry
    source_positions = geometry.src_position(geometry.angles)
    print(source_positions)

    # Visualize source positions in 3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter(
        source_positions[:, 0], 
        source_positions[:, 1],
        source_positions[:, 2]
    )
    fig.colorbar(img)

    plt.show()
    plt.savefig('test')