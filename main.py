import argparse
import yaml
import torch
import os
import numpy as np

# Import training and evaluation functions.
from train import training_pipeline
from evaluate import (
    set_global_seed,
    load_data,
    load_model_checkpoint,
    evaluate_model,
    plot_results
)

def load_config(config_path):
    """
    Load configuration from a YAML file and update the device.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config["DEVICE"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return config

def run_training(config, args):
    """
    Run the training pipeline with configuration overrides.
    """
    # Override config values if provided.
    if args.dataset:
        config["DATASET"] = args.dataset
    if args.missing_patterns:
        config["MISSING_PATTERNS"] = args.missing_patterns
    if args.r_m is not None:
        config["R_M"] = args.r_m
    if args.r_mi:
        config["R_MI"] = args.r_mi
    if args.output_csv:
        config["OUTPUT_CSV"] = args.output_csv

    print("Starting training...")
    training_pipeline(config)

def run_evaluation(config, args):
    """
    Run evaluation on the test set using a checkpoint.
    
    If --checkpoint is not provided, the default checkpoint is automatically 
    selected from the folder "checkpoints/SIGFormer_{dataset}" using the default
    missing pattern (first in config["MISSING_PATTERNS"]) and missing ratio (config["R_M"]).
    
    Requires:
      --unknown_ratio: Ratio for unknown sensors (default 0.25)
      --output_dir: Directory to save evaluation plots.
    """
    # Override config values if provided.
    if args.dataset:
        config["DATASET"] = args.dataset

    device = config["DEVICE"]
    # Set seed for reproducibility.
    set_global_seed(config.get("GLOBAL_SEED", 1))
    
    # Load dataset.
    dataset = config["DATASET"]
    A, X, training_set, val_set, test_set, full_set = load_data(dataset)
    
    # Define unknown sensor set based on unknown_ratio.
    num_unknown = int(args.unknown_ratio * test_set.shape[1])
    # Using sorted order for reproducibility.
    unknown_set = set(sorted(range(test_set.shape[1]))[:num_unknown])
    
    E_maxvalue = config["E_MAXVALUE_DICT"].get(dataset, 1)
    h = config["H"]
    z = config["Z"]
    K = config["K"]

    # If checkpoint not provided, construct default checkpoint path.
    if not args.checkpoint:
        default_pattern = config["MISSING_PATTERNS"][0]
        default_r_m = config["R_M"]
        checkpoint_folder = os.path.join("checkpoints", f"SIGFormer_{dataset}")
        default_ckpt = f"best_model_p={default_pattern}_r_m={default_r_m:.2f}.pth"
        args.checkpoint = os.path.join(checkpoint_folder, default_ckpt)
        print(f"No checkpoint provided. Using default checkpoint: {args.checkpoint}")
    
    # Check that the checkpoint file exists.
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Load the trained model.
    model = load_model_checkpoint(args.checkpoint, h, z, K, device)
    
    print("Evaluating model on test set...")
    reconstructed = evaluate_model(model, A, test_set, unknown_set, E_maxvalue, device)
    
    # Compute basic error metrics.
    gt = test_set[:reconstructed.shape[0]]
    mae = float(np.mean(np.abs(gt - reconstructed)))
    rmse = float(np.sqrt(np.mean((gt - reconstructed) ** 2)))
    print(f"Evaluation Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Plot results for a few sensors from the unknown set.
    sensors_to_plot = list(unknown_set)[:3]  # Plot up to three sensors.
    for sensor in sensors_to_plot:
        plot_results(gt, reconstructed, sensor, args.output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Run training or evaluation for the SIGFormer model."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--dataset", type=str,
                        help="Dataset name (e.g., 'pems_flow', 'metr', etc.)")
    parser.add_argument("--missing_patterns", nargs="+",
                        help="List of missing patterns (allowed: cs, s, t, b, r)")
    parser.add_argument("--r_m", type=float,
                        help="Missing ratio (allowed options: 0.2, 0.4, 0.6, 0.8)")
    parser.add_argument("--r_mi", type=float,
                        help="Ratio for the unknown sensor set")
    parser.add_argument("--output_csv", type=str,
                        help="Output CSV filename for training results")
    
    # Evaluation-specific arguments.
    parser.add_argument("--test", action="store_true",
                        help="If set, run evaluation instead of training.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to the trained model checkpoint (.pth) for evaluation. If not provided, the default checkpoint from config will be used.")
    parser.add_argument("--unknown_ratio", type=float, default=0.25,
                        help="Ratio for unknown sensors in evaluation.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation plots.")
    
    args = parser.parse_args()
    
    # Load configuration.
    config = load_config(args.config)
    
    if args.test:
        run_evaluation(config, args)
    else:
        run_training(config, args)

if __name__ == "__main__":
    main()
