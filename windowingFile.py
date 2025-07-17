import mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # For progress tracking

def collect_edf_files(directory):
    epilepsyPaths = []
    for child in os.listdir(directory):
        child_path = os.path.join(directory, child)
        if os.path.isdir(child_path):
            epilepsyPaths.extend(collect_edf_files(child_path))  # Recurse into subdirectories
        elif child.endswith(".edf"):
            epilepsyPaths.append(child_path)
    return epilepsyPaths

# Start from the parent directory
parentDir = r"file path here"  # Replace with your actual path
EpilipsyPath = collect_edf_files(parentDir)

def preprocess_eeg_data(file_path, sampling_rate=256, window_size_sec=4, overlap_percent=50, n_channels=25):
    # Calculate window size and overlap in samples
    window_size = window_size_sec * sampling_rate  # Convert window size to samples
    overlap = window_size * overlap_percent // 100  # Calculate overlap in samples

    # Load EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Resample if needed
    if raw.info['sfreq'] != sampling_rate:
        raw.resample(sampling_rate)

    # Apply filters
    raw.notch_filter(50)  # Remove power line noise
    raw.filter(0.5, 40)   # Bandpass filter

    # Get the data
    data = raw.get_data()

    # Select only the first `n_channels` (can be modified to select specific channels)
    if len(data) > n_channels:
        data = data[:n_channels, :]

    # Calculate number of windows
    n_samples = data.shape[1]
    stride = window_size - overlap
    n_windows = ((n_samples - window_size) // stride) + 1

    # Initialize array for windows
    windows = np.zeros((n_windows, data.shape[0], window_size))

    # Create overlapping windows
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[:, start_idx:end_idx]

    # Basic preprocessing for each window
    processed_windows = []
    for window in windows:
        # Remove mean
        window = window - np.mean(window, axis=1, keepdims=True)

        # Standardize
        window = (window - np.mean(window)) / (np.std(window) + 1e-8)
        processed_windows.append(window)

    # Convert processed windows to a numpy array
    processed_windows = np.array(processed_windows)

    return processed_windows

epiData = [preprocess_eeg_data(i) for i in EpilipsyPath[:500]]

# Save the processed data
save_path = r"add your path here"
np.save(save_path, epiData)
print(f"Array saved to {save_path}")
