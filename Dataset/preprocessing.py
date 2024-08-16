import copy

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def scale_adj(A):
    """
    Ref: https://stats.stackexchange.com/questions/565857/weighted-adjacency-matrix-normalization-for-gcn-how-to-normalize-what-about-se
    Scale the weights of the adj matrix. Each matrix i is scaled in the range 
    [1/(max_i-min_i+1), 1], where max_i, min_i - max and min values of matrix i
    S - number of samples, N - number of nodes
    Args:
        A: Array of weighted adjacency matrices (S x N x N) 
    Returns:
        A_scaled: Scaled adjacency matrices (S x N x N)
    """
    # Original -> ([min, max])
    A1 = copy.deepcopy(A) 

    # Ignore zero elements
    zero_mask = A1 == 0 
    A1_masked = np.ma.masked_array(A1, mask=zero_mask)

    # Extract min per matrix ignoring zero elements
    min_vals = np.min(A1_masked, axis=(1, 2), keepdims=True)

    # Shift by min+1 -> ([1, max-min+1])
    A1_masked = A1_masked - min_vals + 1 
    # Extract max per matrix ignoring zero elements
    max_vals = np.max(A1_masked, axis=(1, 2), keepdims=True)
    # Scaled -> [1/(max-min+1), 1] 
    A1_masked = A1_masked / max_vals 
    
    return A1_masked.data


def pad_ohe_features(x, num_nodes):
    """Add OHE identifier of each node to the array of features
    S - number of samples, N - number of nodes, M - number of features
    Args:
        x: Array of features (S x N x M)
        num_nodes: Number of nodes
    Returns:
        x_ohe: Array of features with OHE identifiers (S x N x M+N)
    """ 
    # Create an OHE identifiers for one sample
    ohe = np.eye(num_nodes)
    # Add OHE features for each sample 
    ohe_features = np.stack([ohe]* x.shape[0])
    # Concatenate the OHE features
    x_ohe = np.concatenate((x, ohe_features), axis=-1)
    
    return x_ohe
    
    
def scale_feature(x, feature_id, train_mask, val_mask, test_mask):
    """
    Scale the selected feature in the dataset
    Args:
        x: Array of features (S x N x M)
        feature_id: ID of the feature to scale
        train_mask: Boolean mask for training data
        val_mask: Boolean mask for validation data
        test_mask: Boolean mask for test data
    Returns:
        x: Scaled array of features (S x N x M) (feature_id is scaled)
    """
    # Train the scaler on training and validation
    scaler = StandardScaler()
    
    # Flatten the feature wrt. the nodes
    scaler.fit(x[(train_mask | val_mask), :, feature_id].reshape(-1, 1)) # flatten
    
    # apply the scaler to the data (prevent information leak to the test data)
    x[train_mask, :, feature_id:feature_id+1] = scaler.transform(
        x[train_mask, :, feature_id].reshape(-1, 1)
        ).reshape(x[train_mask].shape[0], x[train_mask].shape[1],  1)
    x[val_mask, :, feature_id:feature_id+1] = scaler.transform(
        x[val_mask, :, feature_id].reshape(-1, 1)
        ).reshape(x[val_mask].shape[0], x[val_mask].shape[1], 1)
    x[test_mask, :, feature_id:feature_id+1] = scaler.transform(
        x[test_mask, :, feature_id].reshape(-1, 1)
        ).reshape(x[test_mask].shape[0], x[test_mask].shape[1], 1)
    
    return x

def get_train_val_test_masks(num_samples, y=None):
    """
    Get the train, validation and test masks stratified by y.
    Args:
        num_samples: Number of samples
        y: Array of labels (optional)
    Returns:
        train_mask, val_mask, test_mask: Boolean masks for training, validation and test data
    """
    # Get the indices of the training, validation and test samples
    id_train, id_test_ = train_test_split(np.arange(num_samples), test_size=0.3, stratify=y, random_state=0, shuffle=True)
    id_val, id_test = train_test_split(np.arange(id_test_.shape[0]), test_size=0.5, stratify=y[id_test_], random_state=0, shuffle=True)

    print(f"Number of samples: Training {id_train.shape[0]} | Validation {id_val.shape[0]} | Testing {id_test.shape[0]}")        
    
    # Convert them to boolean masks
    train_mask = _get_mask(id_train, num_samples)
    val_mask = _get_mask(id_val, num_samples)
    test_mask = _get_mask(id_test, num_samples)

    return train_mask, val_mask, test_mask


def _get_mask(idx, length):
    """
    Convert an array of indices into a mask
    Args:
        idx: Array of indices
        length: Length of the mask
    Returns:
        msk: Boolean mask
    """
    msk = np.zeros(length, dtype=np.bool_)
    msk[idx] = True
    
    return msk