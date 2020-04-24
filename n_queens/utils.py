import pprint


def print_board(solution):
    board_size = len(solution)
    board = [['–' for _ in range(board_size)] for _ in range(board_size)]
    for idx in range(board_size):
        board[solution[idx]][idx] = '*'
    pprint.pprint(board)
