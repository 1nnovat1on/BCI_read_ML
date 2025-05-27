import os
import yaml
import json
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes
from brainflow.board_shim import BoardShim, BoardIds # To get sampling rate and channel info

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_raw_data_files(base_dir, participant_id=None):
    """
    Finds raw data files (.json) in the specified base directory.
    Args:
        base_dir (str): The base directory for experiment data.
        participant_id (str, optional): Specific participant ID to search for. 
                                        If None, searches all participant directories.
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'filepath' (str): Path to the raw data file.
              'participant_id' (str): Participant ID extracted from the directory structure.
              'word' (str): Word extracted from the filename.
    """
    raw_files_info = []
    if participant_id:
        search_dirs = [os.path.join(base_dir, participant_id)]
        if not os.path.isdir(search_dirs[0]):
            print(f"Warning: Participant directory not found: {search_dirs[0]}")
            return []
    else:
        search_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and d != "preprocessed"] # Avoid preprocessed dir

    for user_dir in search_dirs:
        current_participant_id = os.path.basename(user_dir)
        if not os.path.isdir(user_dir):
            continue
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(user_dir, filename)
                try:
                    word = filename.replace(f"{current_participant_id}_", "").replace(".json", "")
                    raw_files_info.append({
                        'filepath': filepath,
                        'participant_id': current_participant_id,
                        'word': word
                    })
                except Exception as e:
                    print(f"Could not parse filename {filename}: {e}")
    return raw_files_info

def main():
    """Main preprocessing script execution."""
    config = load_config()
    daq_config = config['data_acquisition']
    proc_config = config['data_processing']
    
    experiment_base_dir = daq_config['experiment_base_dir']
    board_id = daq_config['board_id']

    print(f"Scanning for raw data in: {experiment_base_dir}")
    raw_data_files = find_raw_data_files(experiment_base_dir)

    if not raw_data_files:
        print("No raw data files found. Exiting.")
        return

    print(f"Found {len(raw_data_files)} raw data files to process.")

    try:
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        if not eeg_channels:
            print(f"Error: No EEG channels found for board_id {board_id}.")
            return
        target_eeg_channel_index_in_board_data = eeg_channels[0]
        print(f"Using Board ID: {board_id}, Sampling Rate: {sampling_rate} Hz")
        print(f"EEG Channels available: {eeg_channels}. Processing channel: {target_eeg_channel_index_in_board_data}")
    except Exception as e:
        print(f"Error getting board information for board_id {board_id}: {e}")
        return

    all_epochs_data = []
    all_labels = []
    all_participant_ids = []

    # Epoching parameters
    epoch_tmin_s = proc_config['epoch_tmin_s']
    epoch_tmax_s = proc_config['epoch_tmax_s']
    target_epoch_length_samples = int((epoch_tmax_s - epoch_tmin_s) * sampling_rate)

    for file_info in raw_data_files:
        filepath = file_info['filepath']
        participant_id = file_info['participant_id']
        word = file_info['word']
        print(f"\nProcessing: {filepath} (Participant: {participant_id}, Word: {word})")

        try:
            with open(filepath, 'r') as f:
                raw_eeg_data_list = json.load(f)
            
            raw_eeg_data_np = np.array(raw_eeg_data_list)
            
            if raw_eeg_data_np.ndim < 2 or target_eeg_channel_index_in_board_data >= raw_eeg_data_np.shape[0]:
                print(f"Warning: Data in {filepath} is not suitable (shape: {raw_eeg_data_np.shape}, target_idx: {target_eeg_channel_index_in_board_data}). Skipping.")
                continue
            
            single_channel_data = raw_eeg_data_np[target_eeg_channel_index_in_board_data, :].astype(np.float64)
            # print(f"  Original data shape (all channels, samples): {raw_eeg_data_np.shape}")
            # print(f"  Selected channel ({target_eeg_channel_index_in_board_data}) data shape: {single_channel_data.shape}")

            # Filtering
            DataFilter.perform_bandpass(single_channel_data, sampling_rate, proc_config['low_cut_hz'], 
                                        proc_config['high_cut_hz'], proc_config['filter_order'], 
                                        FilterTypes.BUTTERWORTH.value, 0)
            notch_bandwidth_hz = 2.0 
            DataFilter.perform_bandstop(single_channel_data, sampling_rate, proc_config['notch_freq_hz'], 
                                        notch_bandwidth_hz, proc_config['filter_order'], 
                                        FilterTypes.BUTTERWORTH.value, 0)
            # print(f"  Data shape after filtering: {single_channel_data.shape}")

            # Epoching
            event_sample = 0 # Assuming word onset is at the start of the recording
            epoch_start_sample = event_sample + int(epoch_tmin_s * sampling_rate)
            epoch_end_sample = event_sample + int(epoch_tmax_s * sampling_rate)
            
            # Extract the epoch
            extracted_epoch = single_channel_data[max(0, epoch_start_sample) : min(len(single_channel_data), epoch_end_sample)]
            # print(f"  Desired epoch sample range: [{epoch_start_sample}, {epoch_end_sample}]")
            # print(f"  Extracted epoch raw length: {len(extracted_epoch)}")

            # Pad if shorter than target length
            if len(extracted_epoch) < target_epoch_length_samples:
                padding_needed = target_epoch_length_samples - len(extracted_epoch)
                # If epoch_start_sample was negative and cut, part of the padding might conceptually be at the start.
                # However, simple end-padding is common.
                extracted_epoch = np.pad(extracted_epoch, (0, padding_needed), 'constant', constant_values=(0,))
                # print(f"  Padded epoch to length: {len(extracted_epoch)}")
            elif len(extracted_epoch) > target_epoch_length_samples: # Should not happen if epoch_end_sample is calculated correctly
                extracted_epoch = extracted_epoch[:target_epoch_length_samples]
                # print(f"  Truncated epoch to length: {len(extracted_epoch)}")


            if len(extracted_epoch) != target_epoch_length_samples:
                print(f"  Warning: Epoch for {word} from {participant_id} has unexpected length {len(extracted_epoch)} after padding/truncation. Skipping.")
                continue

            # Normalization
            if proc_config['normalization_method'] == "zscore_channel":
                mean = np.mean(extracted_epoch)
                std = np.std(extracted_epoch)
                if std > 1e-6: # Avoid division by zero or very small std
                    processed_epoch = (extracted_epoch - mean) / std
                else:
                    processed_epoch = extracted_epoch # Or np.zeros_like(extracted_epoch)
                # print(f"  Normalized epoch (z-score). Mean: {np.mean(processed_epoch):.2f}, Std: {np.std(processed_epoch):.2f}")
            elif proc_config['normalization_method'] == "none":
                processed_epoch = extracted_epoch
            else:
                print(f"Warning: Unknown normalization method '{proc_config['normalization_method']}'. Skipping normalization.")
                processed_epoch = extracted_epoch
            
            all_epochs_data.append(processed_epoch)
            all_labels.append(word)
            all_participant_ids.append(participant_id)
        
        except FileNotFoundError:
            print(f"Error: File not found {filepath}. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filepath}: {e}")
            import traceback
            traceback.print_exc()


    if not all_epochs_data:
        print("\nNo epochs were successfully processed. Exiting.")
        return

    # Convert lists to NumPy arrays
    X_data = np.array(all_epochs_data)
    # Reshape X_data to (num_epochs, 1, sequence_length) for BrainWaveNet
    X_data = X_data.reshape((X_data.shape[0], 1, X_data.shape[1]))
    
    y_labels = np.array(all_labels)
    participant_ids_np = np.array(all_participant_ids)

    print(f"\nProcessed {len(all_epochs_data)} epochs.")
    print(f"  X_data shape: {X_data.shape}")
    print(f"  y_labels shape: {y_labels.shape}")
    print(f"  participant_ids shape: {participant_ids_np.shape}")

    # Save preprocessed data
    preprocessed_dir = os.path.join(experiment_base_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    save_path = os.path.join(preprocessed_dir, "preprocessed_eeg_data.npz")
    np.savez(save_path, 
             X_data=X_data, 
             y_labels=y_labels, 
             participant_ids=participant_ids_np,
             sampling_rate=sampling_rate) # Also save sampling rate, might be useful

    print(f"\nPreprocessed data saved to: {save_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
