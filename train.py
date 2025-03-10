import torch
import numpy as np
import torch.optim as optim
from torch import nn
import os
import time
import datetime
import csv
import random
from itertools import product
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_sigformer import SIGFormer
from generate_mask import generate_mask
from utils import *  # Ensure your utils.py defines functions like load_metr_la_rdata, load_pems_rdata, calculate_random_walk_matrix

def set_global_seed(seed):
    """
    Set the global random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Uncomment if CUDA reproducibility is needed:
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

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

    # Split data: 60% training, 20% validation, 20% test.
    split_line1 = int(X.shape[1] * 0.6)
    split_line2 = int(X.shape[1] * 0.8)

    training_set = X[:, :split_line1].transpose()
    val_set = X[:, split_line1:split_line2].transpose()
    test_set = X[:, split_line2:].transpose()

    full_set = set(range(X.shape[0]))
    return A, X, training_set, val_set, test_set, full_set

def get_unknown_set(full_set, r_mi):
    """
    Generate the unknown sensor set based on ratio r_mi.
    
    Parameters:
        full_set (set): Set of all sensor IDs.
        r_mi (float): Ratio of unknown sensor IDs.
        
    Returns:
        unknow_set (set): Randomly selected unknown sensor IDs.
    """
    total_sensors = len(full_set)
    num_unknown = int(r_mi * total_sensors)
    unknow_set = set(random.sample(list(full_set), num_unknown))
    return unknow_set

def test_error(STmodel, unknow_set, data_set, A_s, Missing0, E_maxvalue, device):
    """
    Compute error metrics on the given dataset.
    
    Parameters:
        STmodel: Model instance.
        unknow_set (set): Set of unknown sensor indices.
        data_set (np.ndarray): Validation or test data.
        A_s: Adjacency matrix.
        Missing0 (bool): Whether to treat zeros as missing.
        E_maxvalue (float): Scaling factor for normalization.
        device: Torch device.
    
    Returns:
        MAE, RMSE, MAPE: Error metrics.
        o: Imputed output.
        truth: Ground truth.
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    omask = np.ones(data_set.shape)
    if Missing0:
        omask[data_set == 0] = 0
    test_inputs = (data_set * omask).astype('float32')

    missing_index = np.ones(np.shape(data_set))
    missing_index[:, list(unknow_set)] = 0

    o = np.zeros([data_set.shape[0] // time_dim * time_dim, test_inputs.shape[1]])

    for i in range(0, data_set.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs[i:i + time_dim, :]
        missing_inputs = missing_index[i:i + time_dim, :]
        T_inputs = inputs * missing_inputs
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32')).to(device)
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32')).to(device)

        imputation = STmodel(T_inputs, A_q, A_h)
        # Convert imputation to numpy adaptively based on device.
        if device.type == "cuda":
            imputation = imputation.cuda().data.cpu().numpy()
        else:
            imputation = imputation.data.numpy()
            
        o[i:i + time_dim, :] = imputation[0, :, :]

    o = o * E_maxvalue
    truth = test_inputs[0:data_set.shape[0] // time_dim * time_dim]
    o[missing_index[0:data_set.shape[0] // time_dim * time_dim] == 1] = truth[missing_index[0:data_set.shape[0] // time_dim * time_dim] == 1]

    test_mask = 1 - missing_index[0:data_set.shape[0] // time_dim * time_dim]
    if Missing0:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) ** 2) / np.sum(test_mask))
    MAPE = np.sum(np.abs(o - truth) / (truth + 1e-5)) / np.sum(test_mask)

    return MAE, RMSE, MAPE, o, truth

class loss_function(nn.Module):
    """
    Balanced loss for observed and missing value reconstruction.
    """
    def __init__(self, alpha=0.2):
        """
        Initialize the loss function.
        
        Parameters:
            alpha (float): Weight for the observed value loss.
        """
        super(loss_function, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets, mask):
        """
        Compute the balanced loss.
        
        Parameters:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth.
            mask (torch.Tensor): Binary mask for observed values.
            
        Returns:
            torch.Tensor: Loss value.
        """
        observed_mask = mask
        missing_mask = 1 - mask
        observed_loss = self.mse_loss(predictions * observed_mask, targets * observed_mask)
        missing_loss = self.mse_loss(predictions * missing_mask, targets * missing_mask)
        balanced_loss = self.alpha * observed_loss + (1 - self.alpha) * missing_loss
        return balanced_loss

def training_pipeline(config):
    """
    Training pipeline that iterates over missing patterns using a single missing ratio.
    
    Parameters:
        config (dict): Dictionary containing all configuration parameters.
    """
    set_global_seed(config["GLOBAL_SEED"])
    
    # Unpack parameters from config.
    h = config["H"]
    z = config["Z"]
    K = config["K"]
    n_o_n_m = config["N_O_N_M"]
    max_iter = config["MAX_ITER"]
    learning_rate = config["LEARNING_RATE"]
    batch_size = config["BATCH_SIZE"]
    device = config["DEVICE"]
    LOSS_ALPHA = config["LOSS_ALPHA"]
    ETA_MIN = config["ETA_MIN"]
    E_MAXVALUE_DICT = config["E_MAXVALUE_DICT"]
    dataset = config["DATASET"]
    output_csv = config["OUTPUT_CSV"]

    # Use the single missing ratio.
    r_m = config["R_M"]

    print("Using device:", device)
    E_maxvalue = E_MAXVALUE_DICT[dataset]

    # Load dataset splits.
    A, X, training_set, val_set, test_set, full_set = load_data(dataset)
    unknow_set = get_unknown_set(full_set, config["R_MI"])
    know_set = full_set - unknow_set
    print(f"Fixed unknown set size: {len(unknow_set)}, Known set size: {len(know_set)}")

    # Create a top-level "checkpoints" folder if it doesn't exist.
    checkpoint_base = "checkpoints"
    os.makedirs(checkpoint_base, exist_ok=True)
    # Create a descriptive folder under checkpoints without the timestamp.
    results_folder = os.path.join(checkpoint_base, f"SIGFormer_{dataset}")
    os.makedirs(results_folder, exist_ok=True)

    # Prepare CSV logging.
    csv_headers = ["missing_pattern", "r_m", "best_epoch", "best_mae", "best_rmse", "best_mape", "model_path"]
    csv_data = []

    # Iterate over missing patterns (each will use the same r_m value).
    for missing_pattern in config["MISSING_PATTERNS"]:
        print(f"Training with pattern='{missing_pattern}', r_m={r_m:.2f}...")
        n_m = int(r_m * n_o_n_m)
        print(f"Calculated n_m={n_m} for training.")

        # Define the model and training components.
        STmodel = SIGFormer(h, z, K, L=1, heads=4, heads_temp=1, ff_hidden_dim=128)
        STmodel.to(device)
        criterion = loss_function(alpha=LOSS_ALPHA)
        optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate, weight_decay=0)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=ETA_MIN)

        best_mae = float('inf')
        best_epoch = None
        best_rmse = None
        best_mape = None
        best_model_path = None

        # Training loop.
        for epoch in range(max_iter):
            start_time = time.time()
            for i in range(training_set.shape[0] // (h * batch_size)):
                t_random = np.random.randint(0, high=(training_set.shape[0] - h), size=batch_size, dtype='l')
                know_mask = set(random.sample(range(0, len(know_set)), n_o_n_m))
                feed_batch = []
                for j in range(batch_size):
                    feed_batch.append(training_set[t_random[j]: t_random[j] + h, :][:, list(know_mask)])
                inputs = np.array(feed_batch)
                A_dynamic = A[list(know_mask), :][:, list(know_mask)]
                mask_matrix = [generate_mask(batch, A_dynamic, r_m, missing_pattern) for batch in inputs]
                mask_matrix = np.array(mask_matrix)
                mask_matrix_ = torch.from_numpy(mask_matrix).to(device)
                inputs_omask = np.ones(np.shape(inputs))
                if dataset != 'NREL':
                    inputs_omask[inputs == 0] = 0

                Mf_inputs = inputs * inputs_omask * mask_matrix / E_maxvalue
                Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
                mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)
                A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
                A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)
                outputs = torch.from_numpy(inputs / E_maxvalue).to(device)

                STmodel.train()
                optimizer.zero_grad()
                X_res = STmodel(Mf_inputs, A_q, A_h)
                loss = criterion(X_res * mask, outputs * mask, mask_matrix_)
                loss.backward()
                optimizer.step()

            scheduler.step()
            current_lr = scheduler.get_lr()[0]
            # Evaluate on validation set.
            STmodel.eval()
            MAE_v, RMSE_v, MAPE_v, pred, truth = test_error(STmodel, unknow_set, val_set, A, True, E_maxvalue, device)

            if MAE_v < best_mae:
                best_mae = MAE_v
                best_rmse = RMSE_v
                best_mape = MAPE_v
                best_epoch = epoch
                best_model_path = os.path.join(results_folder, f"best_model_p={missing_pattern}_r_m={r_m:.2f}.pth")
                torch.save(STmodel.state_dict(), best_model_path)

            if epoch % 10 == 0 or epoch == max_iter - 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}: MAE={MAE_v:.4f}, RMSE={RMSE_v:.4f}, MAPE={MAPE_v:.4f}, Time={elapsed:.2f}s, LR={current_lr:.6f}")

        print(f"Best MAE={best_mae:.4f}, Best RMSE={best_rmse:.4f}, Best MAPE={best_mape:.4f} at epoch={best_epoch}.")
        print(f"Saved model checkpoint to {best_model_path}")
        csv_data.append([missing_pattern, r_m, best_epoch, best_mae, best_rmse, best_mape, best_model_path])

    # Write CSV results.
    csv_file = os.path.join(results_folder, output_csv)
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)
    print(f"Training results saved to {csv_file}")
