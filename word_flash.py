import os
import time
import json
import yaml # Import YAML library
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# Function to load YAML configuration
def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_recording(participant_id, word, data, base_dir, file_extension='.json'):
    """Save the neural recording data for a given participant and word."""
    user_dir = os.path.join(base_dir, participant_id)
    os.makedirs(user_dir, exist_ok=True)
    filename = f"{participant_id}_{word}{file_extension}"
    filepath = os.path.join(user_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data.tolist(), f)  # Ensure conversion to list if using numpy array

def flash_and_record(participant_id, word, board, recording_duration, base_dir):
    """Flash the word and record the corresponding brainwave data."""
    print(f"Flashing word: {word}")
    # Here, integrate with a GUI or visual presentation system if desired
    time.sleep(1)  # Simulate the time required for word display
    try:
        board.start_stream()
        time.sleep(recording_duration)
        data = board.get_board_data()
        board.stop_stream()
        save_recording(participant_id, word, data, base_dir=base_dir)
        print(f"Data recorded for word: {word}")
    except BrainFlowError as e:
        print(f"BrainFlowError during streaming: {e}")
        # Optionally, decide if you want to try to stop/release session here or let main handle it
        raise # Re-raise the exception to be caught by main or to stop the script

def main():
    # Load configuration
    config = load_config()
    daq_config = config['data_acquisition']
    
    board_id = daq_config['board_id']
    recording_duration_s = daq_config['recording_duration_s']
    inter_trial_interval_s = daq_config['inter_trial_interval_s']
    experiment_base_dir = daq_config['experiment_base_dir']
    word_list_file = daq_config['word_list_file']

    # Setup BrainFlow parameters
    params = BrainFlowInputParams()
    if 'serial_port' in daq_config and daq_config['serial_port'] is not None:
        params.serial_port = daq_config['serial_port']
    if 'mac_address' in daq_config and daq_config['mac_address'] is not None:
        params.mac_address = daq_config['mac_address']

    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
    except BrainFlowError as e:
        print(f"Failed to prepare BrainFlow session: {e}")
        print("Please check your board connection and configuration.")
        return # Exit if session cannot be prepared

    # Load words from file
    try:
        with open(word_list_file, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        if not words:
            print(f"Word list file '{word_list_file}' is empty or contains no valid words.")
            board.release_session()
            return
        print(f"Loaded {len(words)} words from {word_list_file}")
    except FileNotFoundError:
        print(f"Error: Word list file not found at {word_list_file}")
        board.release_session()
        return
    
    # Get participant ID
    participant_id = input("Enter participant ID (e.g., subject_01): ").strip()
    if not participant_id:
        print("Participant ID cannot be empty.")
        board.release_session()
        return

    print(f"\nStarting experiment for participant: {participant_id}")
    print(f"Recording duration per word: {recording_duration_s}s")
    print(f"Interval between words: {inter_trial_interval_s}s")
    print(f"Data will be saved in: {os.path.join(experiment_base_dir, participant_id)}\n")

    for i, word in enumerate(words):
        print(f"\nTrial {i+1}/{len(words)}")
        try:
            flash_and_record(participant_id, word, board, recording_duration_s, experiment_base_dir)
        except BrainFlowError:
            print(f"Skipping word '{word}' due to previous BrainFlow error during streaming.")
            # Decide if you want to attempt to reconnect or just skip
            # For now, we'll assume the stream is compromised and stop.
            break 
        time.sleep(inter_trial_interval_s)

    try:
        board.release_session()
        print("\nExperiment finished. Session released.")
    except BrainFlowError as e:
        print(f"BrainFlowError during session release: {e}")


if __name__ == "__main__":
    main()
