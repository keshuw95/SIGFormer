import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def process_daily_file(file_path, sensor_ids):
    """
    Process one daily CSV file to compute a matrix of average flow per lane.
    
    Parameters:
      file_path (str): Path to the CSV file.
      sensor_ids (iterable): List or array of sensor (station) IDs to keep.
    
    Returns:
      pd.DataFrame: Daily matrix (shape: (num_sensors, 288)) with average flow per lane.
                    If file processing fails, returns None.
    """
    # Define fixed station columns (first 12 columns)
    station_columns = [
        "Timestamp", "Station", "District", "Freeway #", "Direction of Travel",
        "Lane Type", "Station Length", "Samples", "% Observed", "Total Flow",
        "Avg Occupancy", "Avg Speed"
    ]
    
    try:
        # Load the CSV without headers to determine the total number of columns
        temp_df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    total_columns = temp_df.shape[1]
    lane_columns_count = total_columns - len(station_columns)
    max_number_of_lanes = lane_columns_count // 5
    
    # Create lane-specific column names dynamically
    lane_columns = []
    for i in range(1, max_number_of_lanes + 1):
        lane_columns.extend([
            f"Lane {i} Samples",
            f"Lane {i} Flow",
            f"Lane {i} Avg Occ",
            f"Lane {i} Avg Speed",
            f"Lane {i} Observed"
        ])
    
    # Combine fixed and lane-specific column names
    column_names = station_columns + lane_columns
    
    # Read the CSV with proper column names
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    # Filter to include only sensor IDs we care about
    df = df[df['Station'].isin(sensor_ids)].copy()
    
    # Convert Timestamp to datetime; assuming format "MM/DD/YYYY HH24:MI:SS"
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%m/%d/%Y %H:%M:%S")
    
    # Identify lane flow columns dynamically (those that include "Flow" after "Lane")
    lane_flow_columns = [col for col in df.columns if col.startswith("Lane") and "Flow" in col]
    
    # Count valid lane flows (non-null) for each row
    df['num_lanes'] = df[lane_flow_columns].notna().sum(axis=1)
    
    # Compute average flow per lane (veh/(5min*lane)), safeguard against division by zero
    df['Avg Flow per Lane'] = df.apply(
        lambda row: row['Total Flow'] / row['num_lanes'] if row['num_lanes'] > 0 else np.nan,
        axis=1
    )
    

    # Create a complete time index for the day (288 intervals; 5-minute intervals)
    day = df['Timestamp'].dt.date.iloc[0]
    start_time = pd.Timestamp(f"{day} 00:00:00")
    end_time = pd.Timestamp(f"{day} 23:55:00")
    time_index = pd.date_range(start=start_time, end=end_time, freq="5T")
    
    # Pivot table: rows are Station, columns are Timestamp, value is Avg Flow per Lane.
    daily_pivot = df.pivot_table(
        index='Station',
        columns='Timestamp',
        values='Avg Flow per Lane',
        aggfunc='mean'
    )
    
    # Reindex columns to have the full day (all 288 intervals)
    daily_pivot = daily_pivot.reindex(columns=time_index)
    
    # Fill missing values with 0 (or adjust as needed)
    daily_matrix = daily_pivot.fillna(0)
    
    # Reindex rows to ensure the same order as sensor_ids (fill missing stations with 0)
    daily_matrix = daily_matrix.reindex(sensor_ids, fill_value=0)
    
    return daily_matrix


# Define smoothing function that also smooths out outliers
def smooth_row_with_outliers(row, window_size=5, threshold=2):
    """
    Smooth a 1D array (row) by applying a moving average and
    replacing abrupt outliers. For each index, compute the local median
    (over a sliding window) and, if the deviation is larger than threshold * std,
    replace the original value with the moving average value.
    
    Parameters:
    row         : 1D numpy array of flow data for one sensor.
    window_size : Size of the window for both moving average and local median.
    threshold   : Outlier threshold factor (times the row's standard deviation).
    
    Returns:
    smoothed_row: 1D numpy array after smoothing.
    """
    # Moving average filter
    kernel = np.ones(window_size) / window_size
    row_avg = np.convolve(row, kernel, mode='same')
    
    # Prepare output array (copy of the original)
    smoothed = row.copy()
    
    # Global standard deviation for the row
    row_std = np.std(row)
    
    for i in range(len(row)):
        # Define sliding window indices
        start = max(0, i - window_size // 2)
        end = min(len(row), i + window_size // 2 + 1)
        local_median = np.median(row[start:end])
        
        # If the deviation from the local median is large, replace with moving average value
        if np.abs(row[i] - local_median) > threshold * row_std:
            smoothed[i] = row_avg[i]
    
    return smoothed

if __name__ == "__main__":
    # Load sensor locations to get the 207 sensor IDs.
    sensor_df = pd.read_csv("data/raw_data/pems_flow/graph_sensor_locations.csv")
    sensor_ids = sensor_df['sensor_id'].unique()

    # List to store daily matrices
    daily_matrices = []

    # Define date range: March 1, 2012 to April 30, 2012.
    date_range = pd.date_range(start="2012-03-01", end="2012-04-30", freq="D")

    # Directory containing the raw CSV files.
    data_dir = "data/raw_data/pems_download/"
    print(data_dir)
    for date in date_range:
        # Construct the file name, e.g., d07_text_station_5min_2012_03_01.csv
        file_name = f"d07_text_station_5min_{date:%Y_%m_%d}.csv"
        file_path = os.path.join(data_dir, file_name)
        print(file_path)
        if os.path.exists(file_path):
            daily_matrix = process_daily_file(file_path, sensor_ids)
            if daily_matrix is not None:
                daily_matrices.append(daily_matrix)
            else:
                print(f"Skipping {file_name} due to processing error.")
        else:
            print(f"File not found: {file_name}")

    # Concatenate daily matrices along the time axis
    if daily_matrices:
        # Each daily matrix is of shape (207, 288). Concatenate them horizontally.
        X = pd.concat(daily_matrices, axis=1)
        print("Final matrix X shape:", X.shape)
    else:
        print("No daily matrices were processed successfully.")

    X = X.to_numpy()

    # Apply the smoothing function to each sensor (each row)
    X_smoothed = np.array([smooth_row_with_outliers(row, window_size=5, threshold=1) for row in X])
    print("Applied smoothing filter with outlier correction. New shape:", X_smoothed.shape)


    A = np.load('data/raw_data/pems_flow/adj_mat.npy')

    drop_indices = [136, 190]
    X_ = np.delete(X_smoothed, drop_indices, axis=0)
    A_ = np.delete(A, drop_indices, axis=0)  # drop rows
    A_ = np.delete(A_, drop_indices, axis=1)  # drop columns

    X_ = X_.T
    print("Final X shape", X_.shape, "Final A shape:", A_.shape)
    np.save('data/pems_flow/adj_mat.npy', A_)
    np.save('data/pems_flow/node_values.npy', X_)
