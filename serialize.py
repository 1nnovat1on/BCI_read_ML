import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(10)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    # Convert board data to a pandas DataFrame
    df = pd.DataFrame(np.transpose(data))
    print('Data from the Board:')
    print(df.head(10))

    # Serialize the data using the BrainFlow API
    DataFilter.write_file(data, 'brain_data.csv', 'w')
    restored_data = DataFilter.read_file('brain_data.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))
    print('Data from the File:')
    print(restored_df.head(10))

if __name__ == "__main__":
    main()
