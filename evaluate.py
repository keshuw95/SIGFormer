import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import yaml
from utils import load_metr_la_rdata, load_pems_rdata, calculate_random_walk_matrix
from model_sigformer import SIGFormer

def set_global_seed(seed=1):
    """
    Set the global random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(dataset):
    """
    Load dataset and create train/validation/test splits.
    
    Input:
        dataset (str): Name of the dataset.
        
    Returns:
        A: Adjacency matrix.
        X: Processed data.
        training_set: Training data.
        val_set: Validation data.
        test_set: Test data.
        full_set: Set of all sensor IDs.
    """
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:, 0, :]
    else:
        A, X = load_pems_rdata(dataset)
    
    split_line1 = int(X.shape[1] * 0.6)
    split_line2 = int(X.shape[1] * 0.8)
    
    training_set = X[:, :split_line1].transpose()
    val_set = X[:, split_line1:split_line2].transpose()
    test_set = X[:, split_line2:].transpose()
    
    full_set = set(range(X.shape[0]))
    return A, X, training_set, val_set, test_set, full_set

def load_model_checkpoint(model_path, h, z, K, device):
    """
    Load the trained model from the specified checkpoint.
    
    Parameters:
        model_path (str): Path to the model checkpoint (.pth file).
        h (int): Time dimension.
        z (int): Embedding dimension.
        K (int): Diffusion order.
        device (torch.device): Device to load the model onto.
        
    Returns:
        model (SIGFormer): The loaded model in evaluation mode.
    """
    model = SIGFormer(h, z, K, L=1, heads=4, heads_temp=1, ff_hidden_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, A, test_set, unknown_set, E_maxvalue, device):
    """
    Evaluate the model on the test set using non-overlapping windows.
    
    Parameters:
        model (SIGFormer): The trained model.
        A (np.ndarray): The adjacency matrix.
        test_set (np.ndarray): The test dataset.
        unknown_set (set): Set of sensor indices considered unknown.
        E_maxvalue (float): Scaling factor.
        device (torch.device): Device used for evaluation.
        
    Returns:
        output (np.ndarray): Reconstructed test data.
    """
    unknown_set = set(unknown_set)
    time_dim = model.time_dimension
    # Create observation mask: non-zero values remain 1.
    test_omask = np.ones(test_set.shape)
    test_omask[test_set == 0] = 0
    test_inputs = (test_set * test_omask).astype('float32')
    
    # Create a missing mask: columns corresponding to unknown sensors set to 0.
    missing_index = np.ones(test_set.shape)
    missing_index[:, list(unknown_set)] = 0
    
    output = np.zeros([test_set.shape[0] // time_dim * time_dim, test_inputs.shape[1]])
    
    for i in range(0, test_set.shape[0] // time_dim * time_dim, time_dim):
        window = test_inputs[i:i+time_dim, :]
        window_missing = missing_index[i:i+time_dim, :]
        T_inputs = window * window_missing
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)
        
        A_q = torch.from_numpy((calculate_random_walk_matrix(A).T).astype('float32')).to(device)
        A_h = torch.from_numpy((calculate_random_walk_matrix(A.T).T).astype('float32')).to(device)
        
        imputation = model(T_inputs, A_q, A_h)
        # Adapt conversion based on device.
        if device.type == "cuda":
            imputation = imputation.cuda().data.cpu().numpy()
        else:
            imputation = imputation.data.numpy()
        output[i:i+time_dim, :] = imputation[0, :, :]
    
    output = output * E_maxvalue
    return output

def plot_results(ground_truth, reconstructed, sensor, output_dir):
    """
    Plot ground truth vs reconstructed values for a specified sensor.
    
    Parameters:
        ground_truth (np.ndarray): The ground truth test data.
        reconstructed (np.ndarray): The reconstructed test data.
        sensor (int): Sensor index to plot.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    time_steps = np.arange(ground_truth.shape[0])
    n = min(800, ground_truth.shape[0])
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps[:n], ground_truth[:n, sensor], label="Ground Truth", color="black", linewidth=1.5)
    plt.plot(time_steps[:n], reconstructed[:n, sensor], label="Reconstructed", color="blue", linewidth=1)
    plt.title(f"Sensor {sensor} Reconstruction")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"sensor_{sensor}_reconstruction.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for sensor {sensor} to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the SIGFormer model on the test dataset.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--unknown_ratio", type=float, default=0.25, help="Ratio of sensors to treat as unknown.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation plots.")
    args = parser.parse_args()
    
    # Load configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["DEVICE"] = device
    
    # Set seed for reproducibility.
    set_global_seed(config.get("GLOBAL_SEED", 1))
    
    # Load dataset.
    dataset = config["DATASET"]
    A, X, training_set, val_set, test_set, full_set = load_data(dataset)
    
    # Define unknown sensor set based on provided unknown_ratio.
    num_unknown = int(args.unknown_ratio * test_set.shape[1])
    unknown_set = set(random.sample(range(test_set.shape[1]), num_unknown))
    
    # Get scaling factor.
    E_maxvalue = config["E_MAXVALUE_DICT"].get(dataset, 1)
    
    # Model parameters.
    h = config["H"]
    z = config["Z"]
    K = config["K"]
    
    # Load the trained model.
    model = load_model_checkpoint(args.checkpoint, h, z, K, device)
    
    # Evaluate the model on the test set.
    reconstructed = evaluate_model(model, A, test_set, unknown_set, E_maxvalue, device)
    
    # Compute basic error metrics.
    gt = test_set[:reconstructed.shape[0]]
    mae = np.mean(np.abs(gt - reconstructed))
    rmse = np.sqrt(np.mean((gt - reconstructed) ** 2))
    print(f"Evaluation Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Plot results for a few sensors from the unknown set.
    for sensor in list(unknown_set)[:3]:
        plot_results(test_set, reconstructed, sensor, args.output_dir)

if __name__ == "__main__":
    main()
