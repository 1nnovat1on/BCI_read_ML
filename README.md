# EEG Word Classification Experiment

## Overview

This project implements a pipeline for an EEG-based word classification experiment. It involves:
1.  **Data Collection:** Presenting word stimuli to a participant and recording EEG data using a BrainFlow-compatible device.
2.  **Preprocessing:** Filtering the raw EEG signals, epoching the data around word presentation events, and normalizing it.
3.  **Model Training:** Training a neural network (BrainWaveNet) to classify the words based on the processed EEG data.

## Project Structure

```
.
├── config.yaml                 # Main configuration file for all scripts
├── word_lists/
│   └── default_words.txt       # Default list of words for stimuli
├── word_flash.py               # Script for data collection
├── preprocess_data.py          # Script for EEG data preprocessing
├── train_model.py              # Script for training the BrainWaveNet model
├── BrainWaveNet.py             # Python module defining the BrainWaveNet model
├── experiment_data/            # Default directory for storing all experiment outputs
│   ├── <participant_id>/       # Raw EEG recordings (JSON files) per participant
│   ├── preprocessed/           # Preprocessed data (e.g., preprocessed_eeg_data.npz)
│   └── checkpoints/            # Saved model files (e.g., best_model.pth) and label encoder
└── requirements.txt            # Python dependencies
```

## Setup

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The `config.yaml` file is the central place for all settings for data acquisition, preprocessing, and model training.

Key parameters to review and configure:

*   **`data_acquisition`:**
    *   `board_id`: Crucial. Set this to the ID of your BrainFlow board (e.g., Cyton, Ganglion, Synthetic).
    *   `serial_port` / `mac_address`: Required for certain boards.
    *   `recording_duration_s`: Duration to record EEG for each word.
    *   `experiment_base_dir`: Where to save data. Defaults to `experiment_data`.
    *   `word_list_file`: Path to the word list. Defaults to `word_lists/default_words.txt`. You can create your own word list file and update this path.

*   **`data_processing`:**
    *   `low_cut_hz`, `high_cut_hz`, `notch_freq_hz`: Parameters for bandpass and notch filters.
    *   `epoch_tmin_s`, `epoch_tmax_s`: Time window for epoching around the word stimulus.
    *   `normalization_method`: Method for normalizing epoch data (e.g., `zscore_channel`).

*   **`model_training`:**
    *   `input_channels`, `num_filters`, `lstm_hidden_size`: BrainWaveNet architecture parameters. `input_channels` should typically be 1, as the preprocessing script selects a single channel.
    *   `learning_rate`, `batch_size`, `num_epochs`: Training hyperparameters.
    *   `validation_split`: Proportion of data for the validation set.

*   **`general`:**
    *   `random_seed`: For reproducibility.

## Workflow / How to Run

Make sure you have configured `config.yaml` appropriately before running the scripts.

**1. Data Collection:**

*   **Command:**
    ```bash
    python word_flash.py
    ```
*   The script will prompt you to enter a **participant ID**.
*   Raw EEG data for each word will be saved as JSON files in `experiment_data/<participant_id>/`.

**2. Data Preprocessing:**

*   **Command:**
    ```bash
    python preprocess_data.py
    ```
*   This script processes the raw data found within `experiment_data/<participant_id>/` directories.
*   The processed data (epochs, labels, participant info) will be saved into a single file: `experiment_data/preprocessed/preprocessed_eeg_data.npz`.

**3. Model Training:**

*   **Command:**
    ```bash
    python train_model.py
    ```
*   This script loads the `preprocessed_eeg_data.npz` file.
*   It trains the BrainWaveNet model and saves the following into `experiment_data/checkpoints/`:
    *   `best_model.pth`: State dictionary of the model with the best validation accuracy.
    *   `final_model.pth`: State dictionary of the model at the end of training.
    *   `label_encoder_classes.npy`: The mapping of word labels to numerical classes used by the model.

## BrainFlow Board Configuration

*   The `data_acquisition.board_id` in `config.yaml` **must** be set correctly for your specific EEG device.
*   The default is `22` (BoardIds.SYNTHETIC_BOARD.value), which generates synthetic data and is useful for testing the pipeline.
*   If you are using a physical EEG board (e.g., OpenBCI Cyton, Ganglion), you will need to:
    1.  Find the correct `BoardIds` value from the [BrainFlow documentation](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html).
    2.  Update `board_id` in `config.yaml`.
    3.  You may also need to specify `serial_port` (for dongle-based boards like Cyton) or `mac_address` (for Bluetooth boards like Ganglion) in `config.yaml`. Ensure these are uncommented and correctly set if required by your board.
    4.  Make sure your board is properly connected and drivers (if any) are installed.
