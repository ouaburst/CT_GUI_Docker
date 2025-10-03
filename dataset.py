
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

DETECTOR_WIDTH = 230*0.3

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
        assert data_folder_path.is_dir()
        self.data_folder_path = data_folder_path
        
        assert mode in ['training', 'validation', 'testing']

        assert training_proportion+validation_proportion+testing_proportion == 1

        assert isinstance(load_sinogram, bool)
        self.load_sinogram = load_sinogram

        # data_records_path = data_folder_path.joinpath('ml_ready_data_records_one_rotation.csv')
        data_records_path = data_folder_path.joinpath('ml_ready_data_records.csv')
        assert data_records_path.is_file()
        data_records = pd.read_csv(data_records_path)

        # Split training, validation and testing sets
        unique_trees_indices = data_records['tree_ID'].unique()
        n_trees = len(unique_trees_indices)
        if mode == 'training':
            first_index = 0
            last_index  = math.floor(training_proportion*n_trees)
        elif mode == 'validation':
            first_index = math.floor(training_proportion*n_trees)
            last_index  = math.ceil((training_proportion+validation_proportion)*n_trees)
        else:
            first_index = math.ceil((training_proportion+validation_proportion)*n_trees)
            last_index  = None
        self.dataframe = data_records[data_records['tree_ID'].isin(
            unique_trees_indices[first_index:last_index]
            )]
    
    def __len__(self):
        return self.dataframe.__len__()
        
    def __getitem__(self, index) -> Dict:
        slice_row = self.dataframe.iloc[index]
        specie  = slice_row['specie']
        tree_ID = slice_row['tree_ID']
        disk_ID = slice_row['disk_ID']
        
        sample_path = self.data_folder_path.joinpath(f'{specie}_{tree_ID}_{disk_ID}')
        
        angles = np.load(sample_path.joinpath(f'angles.npy'))
        metadata = dict(json.load(open(sample_path.joinpath(f'metadata.json'))))
        axial_positions = np.load(sample_path.joinpath(f'axial_positions.npy'))
       
        A, A_T = self.make_operators(angles, axial_positions, metadata)
        data_dict = {
                'A' : A,
                'A_T' : A_T,
            }
        if self.load_sinogram:
            data_dict['sinogram'] = np.load(sample_path.joinpath(f'sinogram.npy'))
        return data_dict
    
    def random_indices(self):
        return np.random.permutation(self.__len__())
    
    def make_operators(
        self,
        angles:np.ndarray,
        axial_positions:np.ndarray,
        metadata:Dict,
        local_metadata = None,
        torch = False
        ):
        if local_metadata:
            for key, value in local_metadata.items():
                metadata[key] = value

        if metadata['GEOMETRY_ENGINE'] == 'ODL':
            forward_operator, adjoint_operator = parse_ODL_geometry(
                angles, axial_positions, metadata, torch
            )
        else: 
            raise NotImplementedError
        
        return forward_operator, adjoint_operator

if __name__ == '__main__':
    dataset = MITO(mode='training', load_sinogram=False)
    sample = dataset.__getitem__(0)
    geometry:odl.tomo.ConeBeamGeometry = sample['A'].geometry
    source_positions = geometry.src_position(geometry.angles)
    print(source_positions)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter(
        source_positions[:,0], 
        source_positions[:,1],
        source_positions[:,2])
    fig.colorbar(img)

    plt.show()
    plt.savefig('test')
