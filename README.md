#  🚀SIGFormer: Spatial-Temporal Inductive Graph Transformer

`SIGFormer` is a deep learning model for spatio-temporal forecasting and imputation in transportation networks. By leveraging sparse sensor data—such as that collected from fixed sensors, drones, and mobile vehicles—`SIGFormer`reconstructs missing traffic information with high accuracy. 

This repository implements `SIGFormer` as described in the paper:

***"A Deep Learning Enabled Economical Transportation Informatization Framework with Sparsely Located Sensors"*** 📄

---

## Table of Contents

- [Project Structure](#project-structure)
- [Framework Overview](#framework-overview)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Data](#data)
- [System Requirements & Installation](#system-requirements--installation)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)


---

## Project Structure
```
.
├── config.yaml # Model, training, and dataset parameters. 
├── model_sigformer.py # SIGFormer model definition. 
├── train.py # Training pipeline. 
├── evaluate.py # Evaluation routines. 
├── experiment_pattern.ipynb # Notebook for missing pattern experiments. 
├── sensitivity_analysis.ipynb # Notebook for hyperparameter sensitivity analysis. 
├── utils.py # Data loading and graph construction utilities. 
├── generate_mask.py # Functions for generating missing masks. 
├── good_id.txt # "Good" sensor IDs for experiments. 
├── main.py # Unified entry point for training and evaluation. 
└── README.md # Project documentation.
```
---

## Framework Overview

<p align="center">
  <img src="figures/framework_architecture.png" alt="Framework workflow and model architecture." width="600"/>
</p>

---
## Model Architecture

<p align="center">
  <img src="figures/block_architecture.png" alt="SIGFormer framework architecture." width="600"/>
</p>

**SIGFormer** is composed of several key components:
- **Diffusion Graph Convolution (D_GCN):** Implements a diffusion process using Chebyshev recurrence to capture spatial dependencies.
- **Multi-Head DGCN:** Aggregates outputs from multiple D_GCN layers to capture diverse spatial relationships.
- **Edge-Aware Spatial Attention:** Enhances node features by incorporating learned edge information.
- **Temporal Transformer:** Utilizes multi-head temporal attention and a feed-forward network to capture temporal dependencies.
- **SIGFormerBlock:** Integrates all the above components with residual connections and normalization.

---

## Configuration

All configurable parameters are stored in `config.yaml`. 

---

## Training

The training pipeline in `train.py`:
- Loads the dataset and splits it into training, validation, and test sets.
- Dynamically selects unknown sensor sets based on a specified ratio (`R_MI`).
- Iterates over the missing patterns defined in the configuration.
- Trains the SIGFormer model for a fixed number of epochs.
- Evaluates performance on the validation set and saves the best model checkpoint per missing pattern.
- Logs training results (MAE, RMSE, MAPE) to a CSV file.

### Running Training

Run training via the unified main entry point:
```bash
python main.py --config config.yaml --dataset pems_flow --missing_patterns cs s t b r --r_m 0.25 --r_mi 0.25 --output_csv results.csv
```
Or simply:
```bash
python main.py 
```

---

## Evaluation
The evaluation routines in `evaluate.py`:

- Process the test set using non-overlapping windows.
- Impute missing values using a generated missing mask.
- Compute error metrics (MAE, RMSE, MAPE) to assess model performance.


### Running Evaluation
- Evaluate a trained model by running:
```bash
python main.py --test --dataset pems_flow --unknown_ratio 0.25 --output_dir evaluation_results --checkpoint <path_to_checkpoint>
```
Or simply:
```bash
python main.py --test 
```

---

## Experiments
### Experiment: Missing Pattern Comparison

The `experiment_pattern.ipynb` notebook (and its script version) compares reconstruction performance under different missing patterns (e.g., cs, s, t, b, r) on a selected subset of sensors.

### Experiment: Sensitivity Analysis
The `sensitivity_analysis.ipynb` notebook is provided for interactive sensitivity analysis of training and inference missing ratio.

---

## Data
### PeMS flow data
The dataset comes from PeMS Data Clearinghouse at Caltrans Performance Measurement System (PeMS) (Link: http://pems.dot.ca.gov/).
* `data/pems_flow/node_values.npy`: Flow data
* `data/pems_flow/adj_mat.npy`: Adjacency matrix
* `data/pems_flow/graph_sensor_locations.csv`: Latitude and longitude of sensors

### SeData
SeData is collected by the inductive loop detectors deployed on freeways in Seattle area (Link: https://github.com/zhiyongc/Seattle-Loop-Data).
 * `data/sedata/mat.csv`: Speed data
 * `data/sedata/A.mat`: Adjacency matrix

---

## System Requirements & Installation

### System Requirements:
- Operating System: Windows, macOS, or Linux
- Python 3.7+

### Dependencies
Ensure the following dependencies are installed:
- torch
- numpy
- matplotlib
- seaborn
- scienceplots
- pyyaml
- pandas
- scipy

### Installation Guide:

Install dependencies via pip:

```
pip install torch numpy matplotlib seaborn scienceplots pyyaml pandas scipy
```


---


## Demo 🎥
A small (simulated or real) dataset is provided to demonstrate the software:

### Training Demo:

- Run python `main.py` with default parameters.
- Expected output: Training logs with MAE, RMSE, MAPE values and a CSV log.
- Run time: Approximately 5–15 minutes on a desktop with GPU.

### Evaluation Demo:

- Run `python main.py --test --unknown_ratio 0.25 --output_dir evaluation_results --checkpoint <path_to_checkpoint>`
- Expected output: Evaluation metrics and sample reconstruction plots.
- Run time: Under 5 minutes.


---


## Instructions for Use
1. Data Preparation:
    1. Update `config.yaml` with your dataset paths and parameters if needed.

2. Training:
    1. Run training via: `python main.py`
    2. Monitor training logs and checkpoints saved under the `checkpoints` folder.

3. Evaluation:
    1. Run evaluation via: `python main.py --test --checkpoint <your_model_checkpoint.pth>`
    2. View reconstruction plots in the specified output directory.

4. Experiments:
    1. Use the provided notebooks (`experiment_pattern.ipynb` and `sensitivity_analysis.ipynb`) for interactive experiments and analysis.