import os
import time
import json
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def save_recording(user, word, data, base_dir='recordings', file_extension='.json'):
    """Save the neural recording data for a given user and word."""
    user_dir = os.path.join(base_dir, user)
    os.makedirs(user_dir, exist_ok=True)
    filename = f"{user}_{word}{file_extension}"
    filepath = os.path.join(user_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data.tolist(), f)  # Ensure conversion to list if using numpy array

def flash_and_record(user, word, board, recording_duration=5):
    """Flash the word and record the corresponding brainwave data."""
    print(f"Flashing word: {word}")
    # Here, integrate with a GUI or visual presentation system if desired
    time.sleep(1)  # Simulate the time required for word display
    board.start_stream()
    time.sleep(recording_duration)
    data = board.get_board_data()
    board.stop_stream()
    save_recording(user, word, data)
    print(f"Data recorded for word: {word}")

def main():
    # Setup BrainFlow parameters using a synthetic board (replace with actual board id and parameters as needed)
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()

    # List of 50 words (here, a subset is shown for brevity)
    words = ['trinity', 'tiger', 'bond', 'apple', 'matrix', 'vision', 'neural', 'alpha', 'omega', 'quantum']
    user = "subject_01"

    for word in words:
        flash_and_record(user, word, board, recording_duration=5)
        time.sleep(1)  # Inter-trial interval

    board.release_session()

if __name__ == "__main__":
    main()
