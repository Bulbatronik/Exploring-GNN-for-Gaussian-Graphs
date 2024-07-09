import rdata # to read .RData files
import numpy as np
import pandas as pd

def read_rdata(file_path) -> (np.array, list[pd.DataFrame]): # type: ignore
    """Read a RData file and return an adjusency matrices and data
    Args:
        path (str): path to the RData file    
    Returns:
        adj_matrices (np.array): adjusency matrices for each graph
        data (list[pd.DataFrame]): features and labels for each graph
    """
    parsed_data = rdata.parser.parse_file(file_path)
    converted_data = rdata.conversion.convert(parsed_data)
    
    return converted_data['Adj_matrices'], converted_data['sim_data1']