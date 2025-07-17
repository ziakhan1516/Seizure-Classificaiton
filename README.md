
                              # EEG Workload Classification Using 3D-CNN with Attention
This repository implements a complete pipeline for EEG-based workload (binary) classification using 3D Convolutional Neural Networks combined with a Multi-Head Self-Attention mechanism. The spectrograms are extracted from raw EEG using Short-Time Fourier Transform (STFT).
![EEG 1 drawio](https://github.com/user-attachments/assets/25d0c65a-b133-4da9-bbe6-c3ebdd8b47b7)<?xml version="1.0" encoding="UTF-8"?>

## Dataset Note

The dataset used in this project requires approval before access. It is from the **Temple University Hospital EEG Corpus (TUH EEG)**.  
You need to apply for access here: [http://isip.piconepress.com/projects/tuh_eeg/](http://isip.piconepress.com/projects/tuh_eeg/)

## First, install the dependencies:

```bash
pip install -r requirements.txtt
```
## To run the code
```bash
python eeg_pipeline.py --data_path "data/data_array.npy" --labels_path "data/labels.npy" --save_dir "results" --model_path "results/final_model.h5" --epochs 15 --batch_size 16
```
# The Details are given below:

|   Argument        | Default | Description               |
| ----------------- | ------- | ------------------------- |
| `--window_size`   | `256`   | STFT window size          |
| `--overlap`       | `128`   | STFT overlap              |
| `--sampling_rate` | `256`   | EEG data sampling rate    |
| `--n_channels`    | `21`    | Number of channels to use |
| `--test_size`     | `0.2`   | Test split ratio          |
| `--num_classes`   | `2`     | Number of output classes  |




