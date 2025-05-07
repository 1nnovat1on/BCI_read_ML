import argparse
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, help='Timeout for device discovery or connection', default=0)
    parser.add_argument('--ip-port', type=int, help='IP port', default=0)
    parser.add_argument('--ip-protocol', type=int, help='IP protocol (check IpProtocolType enum)', default=0)
    parser.add_argument('--ip-address', type=str, help='IP address', default='')
    parser.add_argument('--serial-port', type=str, help='Serial port', default='')
    parser.add_argument('--mac-address', type=str, help='MAC address', default='')
    parser.add_argument('--other-info', type=str, help='Other info', default='')
    parser.add_argument('--serial-number', type=str, help='Serial number', default='')
    parser.add_argument('--board-id', type=int, help='Board id (see documentation for supported boards)', required=True)
    parser.add_argument('--file', type=str, help='File path', default='')
    parser.add_argument('--master-board', type=int, help='Master board id for streaming and playback boards', default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port      = args.ip_port
    params.serial_port  = args.serial_port
    params.mac_address  = args.mac_address
    params.other_info   = args.other_info
    params.serial_number= args.serial_number
    params.ip_address   = args.ip_address
    params.ip_protocol  = args.ip_protocol
    params.timeout      = args.timeout
    params.file         = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(10)  # Record for 10 seconds
    data = board.get_board_data()  # Retrieves and clears the buffer
    board.stop_stream()
    board.release_session()
    print(data)

if __name__ == "__main__":
    main()
