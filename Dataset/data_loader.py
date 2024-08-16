import os
import sys
from pathlib import Path

import rdata # to read .RData files
import numpy as np
import pandas as pd

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
    
    # Convert data to numpy array
    return A, data


def get_data(data_path: Path, data_type: str):
    # Check if converted data is already available
    if not os.path.exists(data_path/f'converted/{data_type}'):
        print('Converted data not available, converting...')
        convert_rdata(data_path/f'raw/sim_data_{data_type}.RData')
        
    save_path = os.path.join(Path.cwd() / f'data/preprocessed/{data_type}')
    if os.path.exists(save_path / 'adj_mtrx.npy') and os.path.exists(save_path / 'data.csv'):
        return np.load(save_path / 'adj_mtrx.npy'), pd.read_csv(save_path / 'data.csv')
    
    # If not, convert the .RData file
    file_path = os.path.join(data_path, f'{data_type}.RData')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No RData file found at {file_path}")
    
    return convert_rdata(file_path)
    

def convert_rdata(file_path) -> (np.array, pd.DataFrame): # type: ignore
    """Read a RData file and return an adjacency matrices and data
    Args:
        path (str): path to the RData file    
    Returns:
        adj_matrices (np.array): adjacency matrices for each graph
        data (list[pd.DataFrame]): features and labels for each graph
    """
    assert sys.version_info >= (3, 9), f"This script requires Python 3.9 or later. Current Python version: {'.'.join([str(x) for x in sys.version_info[:3]])}"

    parsed_data = rdata.parser.parse_file(file_path)
    converted_data = rdata.conversion.convert(parsed_data, default_encoding="utf8")
    
    save_path = os.path.join(Path.cwd() / f'data/preprocessed/{data_type}')
    os.makedirs(save_path, exist_ok=True)
    
    # load the datasets
    data_path = os.path.join(Path.cwd() / 'data/raw/')
    
    # store the names of the files in the folder
    data_names = os.listdir(data_path)
    
    for name in data_names:
        file_path = os.path.join(data_path, name)
        
        # get the type of the data
        data_type = name.split('.')[0].split('_')[-1]
        
        save_path = os.path.join(Path.cwd() / f'data/preprocessed/{data_type}')
        os.makedirs(save_path, exist_ok=True)
        
        # load the data fron .RData file
        A, data = read_rdata(file_path)
        print(save_path)
        # save adjacency matrices
        np.save(save_path + '/Adj_mtrx.npy', A)
        # save data
        data.to_csv(save_path + '/data.csv', index=False)
    
    return np.array(converted_data['Adj_matrices']), converted_data['sim_data1']