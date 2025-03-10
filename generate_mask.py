import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils import load_metr_la_rdata

def validate_inputs(X, r_m):
    """
    Validate input parameters.
    
    Parameters:
        X (np.ndarray): Input data array.
        r_m (float): Missing ratio (should be between 0 and 1).
    
    Raises:
        ValueError: If r_m is not in [0,1].
        TypeError: If X is not a NumPy array.
    """
    if not (0 <= r_m <= 1):
        raise ValueError("Missing ratio `r_m` must be between 0 and 1.")
    if not isinstance(X, np.ndarray):
        raise TypeError("Input `X` must be a NumPy array.")

def get_complete_spatial_mask(X, r_m):
    """
    Generate a complete spatial mask where selected nodes are entirely missing over all time steps.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) where 0 indicates missing data.
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    X_MASK = np.ones((T, n))
    total_elements = T * n
    target_missing = int(total_elements * r_m)
    
    mask_count = 0
    selected_nodes = set()
    while mask_count < target_missing and len(selected_nodes) < n:
        i = np.random.randint(n)
        if i not in selected_nodes:
            X_MASK[:, i] = 0  # Mask entire node over all time steps
            selected_nodes.add(i)
            mask_count += T
    return X_MASK

def get_complete_temporal_mask(X, r_m):
    """
    Generate a complete temporal mask where selected time steps are entirely missing across all nodes.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) where 0 indicates missing data.
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    X_MASK = np.ones((T, n))
    total_elements = T * n
    target_missing = int(total_elements * r_m)
    
    mask_count = 0
    selected_times = set()
    while mask_count < target_missing and len(selected_times) < T:
        t_idx = np.random.randint(T)
        if t_idx not in selected_times:
            X_MASK[t_idx, :] = 0  # Mask entire time step across all nodes
            selected_times.add(t_idx)
            mask_count += n
    return X_MASK

def get_spatial_mask(X, r_m, min_window=1, max_window=None):
    """
    Generate a spatial mask that removes data at randomly selected nodes over a random time interval.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
        min_window (int): Minimum time window length.
        max_window (int): Maximum time window length (default: T - 1).
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) with missing entries (0 indicates missing data).
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    if max_window is None:
        max_window = int(T/2)
        
    X_MASK = np.ones((T, n))
    total_elements = T * n
    target_missing = int(total_elements * r_m)
    mask_count = 0

    while mask_count < target_missing:
        # Randomly select a node
        node = np.random.randint(n)
        # Randomly choose a time window
        t_start = np.random.randint(0, T - min_window)
        t_end = np.random.randint(t_start + min_window, min(T, t_start + max_window + 1))
        window_length = t_end - t_start
        X_MASK[t_start:t_end, node] = 0
        mask_count += window_length
    return X_MASK

def get_temporal_mask(X, r_m, min_nodes=1, max_nodes=None):
    """
    Generate a temporal mask that removes data at randomly selected time points for a random set of nodes.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
        min_nodes (int): Minimum number of nodes to mask at a time step.
        max_nodes (int): Maximum number of nodes to mask (default: n).
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) with missing entries (0 indicates missing data).
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    if max_nodes is None:
        max_nodes = int(T/2)
        
    X_MASK = np.ones((T, n))
    total_elements = T * n
    target_missing = int(total_elements * r_m)
    mask_count = 0

    while mask_count < target_missing:
        # Randomly select a time point
        t_idx = np.random.randint(T)
        # Randomly select a set of nodes (non-contiguous selection)
        num_nodes = np.random.randint(min_nodes, max_nodes+1)
        node_indices = np.random.choice(n, size=num_nodes, replace=False)
        X_MASK[t_idx, node_indices] = 0
        mask_count += num_nodes
    return X_MASK

def get_block_mask(X, r_m, A, min_window=1, max_window=None, neighbor_threshold=None):
    """
    Generate a block mask where a node and its neighbors are masked over a random time window.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
        A (np.ndarray): Adjacency matrix of shape (n, n).
        min_window (int): Minimum time window length.
        max_window (int): Maximum time window length (default: T - 1).
        neighbor_threshold (float): Threshold for selecting neighbors (default: mean of A row).
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) with missing entries (0 indicates missing data).
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    if max_window is None:
        max_window = int(T/2)
    if neighbor_threshold is None:
        neighbor_threshold = A.mean()
        
    X_MASK = np.ones((T, n))
    total_elements = T * n
    target_missing = int(total_elements * r_m)
    mask_count = 0

    while mask_count < target_missing:
        # Randomly select a node
        node = np.random.randint(n)
        # Select neighbors based on the threshold
        neighbors = np.where(A[node, :] > neighbor_threshold)[0]
        if len(neighbors) == 0:
            neighbors = np.array([node])  # Fallback to the node itself if no neighbors exceed threshold
        # Randomly select a time window
        t_start = np.random.randint(0, T - min_window)
        t_end = np.random.randint(t_start + min_window, min(T, t_start + max_window + 1))
        window_length = t_end - t_start
        X_MASK[t_start:t_end, neighbors] = 0
        mask_count += len(neighbors) * window_length
    return X_MASK

def get_random_mask(X, r_m):
    """
    Generate a random mask where each data point is independently missing.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        r_m (float): Overall missing ratio.
    
    Returns:
        np.ndarray: Mask matrix of shape (T, n) with missing entries (0 indicates missing data).
    """
    validate_inputs(X, r_m)
    T, n = X.shape
    # Use a vectorized random selection
    X_MASK = np.random.choice([0, 1], size=(T, n), p=[r_m, 1 - r_m])
    return X_MASK

def generate_mask(X, A, r_m, pattern, **kwargs):
    """
    Generate a mask based on the specified missing data pattern.
    
    Parameters:
        X (np.ndarray): Input data of shape (T, n).
        A (np.ndarray): Adjacency matrix of shape (n, n).
        r_m (float): Overall missing ratio.
        pattern (str): Mask pattern type. Options:
            'cs' - Complete Spatial Missing,
            'ct' - Complete Temporal Missing,
            's'  - Spatial Missing,
            't'  - Temporal Missing,
            'b'  - Block Missing,
            'r'  - Random Missing.
        kwargs: Additional keyword arguments for specific mask functions.
        
    Returns:
        np.ndarray: Mask matrix of shape (T, n) where 0 indicates missing data.
    """
    pattern_map = {
        'cs': get_complete_spatial_mask,  # Complete Spatial Missing
        'ct': get_complete_temporal_mask,  # Complete Temporal Missing
        's': get_spatial_mask,             # Spatial Missing
        't': get_temporal_mask,            # Temporal Missing
        'b': lambda X, r_m: get_block_mask(X, r_m, A, **kwargs),  # Block Missing
        'r': get_random_mask               # Random Missing
    }
    
    if pattern not in pattern_map:
        raise ValueError(f"Unsupported pattern '{pattern}'. Choose from {list(pattern_map.keys())}.")
    
    return pattern_map[pattern](X, r_m)

# Main testing block
if __name__ == '__main__':
    t0 = time.time()
    A, X = load_metr_la_rdata()
    # Adjust the shape for testing; assume X originally is (num_nodes, 1, time_steps)
    # We reshape to (time_steps, num_nodes)
    X = X[:, 0, :].T
    r_m = 0.4
    T, n = 5, 20  # For testing: 24 time steps and 20 sensors
    X_test = X[:T, :n]
    A_test = A[:n, :n]

    print("Shape of X_test:", X_test.shape)

    # Define pattern names and corresponding parameters if needed
    pattern_names = {
        'cs': "Complete Spatial Missing",
        'ct': "Complete Temporal Missing",
        's': "Spatial Missing",
        't': "Temporal Missing",
        'b': "Block Missing",
        'r': "Random Missing"
    }

    # For demonstration, select a subset of patterns to visualize
    selected_patterns = {
        'cs': "Complete Spatial Missing",
        'ct': "Complete Temporal Missing",
        's': "Spatial Missing",
        't': "Temporal Missing",
        'b': "Block Missing",
        'r': "Random Missing"
    }

    # Visualize the generated masks for selected patterns
    for pattern, name in selected_patterns.items():
        # For block mask, we pass additional parameters like window length if desired.
        # extra_params = {'min_window': 3, 'max_window': 6} if pattern == 'b' else {}
        X_MASK = generate_mask(X_test, A_test, r_m, pattern)
        plt.figure(figsize=(8, 6))
        sns.heatmap(X_MASK, cbar=False, cmap='viridis')
        plt.title(f"{name} (Missing Ratio: {r_m})")
        plt.xlabel("Sensor ID")
        plt.ylabel("Time Steps")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.show()

    print("Elapsed time:", time.time() - t0)
