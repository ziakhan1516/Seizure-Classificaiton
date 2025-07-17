
                              # EEG Workload Classification Using 3D-CNN with Attention
This repository implements a complete pipeline for EEG-based workload (binary) classification using 3D Convolutional Neural Networks combined with a Multi-Head Self-Attention mechanism. The spectrograms are extracted from raw EEG using Short-Time Fourier Transform (STFT).
![EEG 1 drawio](https://github.com/user-attachments/assets/25d0c65a-b133-4da9-bbe6-c3ebdd8b47b7)<?xml version="1.0" encoding="UTF-8"?>


📦 eeg-workload-classification/
 ┣ 📄 eeg_pipeline.py         # Full pipeline with CLI
 ┣ 📄 requirements.txt        # Dependencies
 ┣ 📄 README.md               # This file

First you have to install the dependensies
pip install -r requirements.txt
For running the code 
python eeg_pipeline.py \
  --data_path "data/data_array.npy" \
  --labels_path "data/labels.npy" \
  --save_dir "results" \
  --model_path "results/final_model.h5" \
  --epochs 15 \
  --batch_size 16
|   Argument        | Default | Description               |
| ----------------- | ------- | ------------------------- |
| `--window_size`   | `256`   | STFT window size          |
| `--overlap`       | `128`   | STFT overlap              |
| `--sampling_rate` | `256`   | EEG data sampling rate    |
| `--n_channels`    | `21`    | Number of channels to use |
| `--test_size`     | `0.2`   | Test split ratio          |
| `--num_classes`   | `2`     | Number of output classes  |

#Citation
@ARTICLE{11017574,
  author={Khan, Ziaullah and Dayal, Aakanksha and Kim, Hee-Cheol},
  journal={IEEE Access},
  title={An Attention-Enhanced 3D-CNN Framework for Spectrogram-Based EEG Analysis in Epilepsy Detection},
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Electroencephalography;Epilepsy;Accuracy;Brain modeling;Feature extraction;Convolutional neural networks;Three-dimensional displays;Time-frequency analysis;Monitoring;Deep learning;EEG signal processing;3D Convolutional Neural Network (3D-CNN);Biomedical signal analysis;Seizure detection;Self-attention mechanism;Short-Time Fourier Transform},
  doi={10.1109/ACCESS.2025.3574646}
}


