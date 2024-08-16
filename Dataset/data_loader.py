import os
import sys
from pathlib import Path
from typing import Tuple

import rdata # to read .RData files
import numpy as np
import pandas as pd


def get_data(data_path: Path, data_type: str)-> Tuple[np.array, pd.DataFrame]:
    """Return the adjacency matrices and the corresponding features
    Args:
        data_path (Path): path to the dataset folder
        data_type (str): type of the data (e.g., 'balanced', 'unbalanced')
    Returns:
        A (np.ndarray): adjacency matrices for the specified data type
        data (pd.DataFrame): features and labels for the specified data type
    """
    # Check if converted data is already available
    if not os.path.exists(data_path/f'converted/{data_type}'):
        print('Converted data not available, converting...')
        convert_rdata(data_path/f'raw/sim_data_{data_type}.RData')
        print('Done')
    
    A, data = load_data(data_path/f'converted/{data_type}')
    return A, data


def load_data(path: Path):
    """Load adjacency matrices from .npy file and data from .csv file
    Args:
        path (str): path to the folder containing adjacency matrices and data
    Returns:
        adj_matrices (np.ndarray): loaded adjacency matrices
        data (pd.DataFrame): loaded data
    """
    # Load adjacency matrices
    A = np.load(f'{path}/Adj_mtrx.npy')
    
    # Load data
    data = pd.read_csv(f'{path}/data.csv')

    return A, data
        

def convert_rdata(file_path) -> Tuple[np.array, pd.DataFrame]: 
    """Read a RData file and return an adjacency matrices and data
    Args:
        path (str): path to the RData file    
    Returns:
        adj_matrices (np.array): adjacency matrices for each graph
        data (list[pd.DataFrame]): features and labels for each graph
    """
    # Required for the 'rdata' module
    assert sys.version_info >= (3, 9), f"This script requires Python 3.9 or later. Current Python version: {'.'.join([str(x) for x in sys.version_info[:3]])}"

    # Extract the necessary data from the RData file
    parsed_data = rdata.parser.parse_file(file_path)
    converted_data = rdata.conversion.convert(parsed_data, default_encoding="utf8")
    A, data = converted_data['Adj_matrices'], converted_data['sim_data1']
    
    # Create a save path for the converted data 
    save_path = os.path.join(Path.cwd() / f'data/preprocessed/{data_type}')
    os.makedirs(save_path, exist_ok=True)
    
    # save adjacency matrices and data
    np.save(save_path + '/Adj_mtrx.npy', A)
    data.to_csv(save_path + '/data.csv', index=False)