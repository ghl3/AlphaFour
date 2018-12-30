#
# A board is represented by a list of lists
#
# board = [[row_bottom, row_1, row_2, ..., row_top],
#           [row1, row2, row3], ...]
#
# The board is always full, but empty spaces are None.
#
# Each non-null element will be one of two values (typically the strings 'R' and 'Y').
#
# The first index picks a column
# The second index picks a row
#
# Boards have 7 columns and 6 rows
#

RED = 'R'
YELLOW = 'Y'

NUM_COLUMNS = 7
NUM_ROWS = 6


def create_board():
    return [[None for _ in range(NUM_ROWS)]
            for _ in range(NUM_COLUMNS)]


def get_top(col):
    for idx, item in enumerate(col):
        if item is None:
            return idx
    return idx


def clone_board(board):
    return [x[:] for x in board]


def clone(board):
    return [[x for x in col] for col in board]


def play(board, column, color):
    """
    Updates the board in place,
    adding the given color piece to
    the given column (dropping it to the top).
    If the column is already full, throws an exception
    """

    board = clone_board(board)

    col = board[column]
    top_idx = get_top(col)
    if top_idx == NUM_ROWS:
        raise Exception("Cannot Play")
    else:
        col[top_idx] = color

    return board


def can_play(board, column):
    col = board[column]
    # If we added one to the top,
    # would it be higher than the allowed
    # number of rows?
    return get_top(col) + 1 < NUM_ROWS


def is_tie(board):
    for i in range(len(board)):
        if can_play(board, i):
            return False
    return True


def is_winner(board, color):
    # check horizontal spaces
    for y in range(NUM_ROWS):
        for x in range(NUM_COLUMNS - 3):
            if board[x][y] == color \
                    and board[x + 1][y] == color \
                    and board[x + 2][y] == color \
                    and board[x + 3][y] == color:
                return True

    # check vertical spaces
    for x in range(NUM_COLUMNS):
        for y in range(NUM_ROWS - 3):
            if board[x][y] == color \
                    and board[x][y + 1] == color \
                    and board[x][y + 2] == color \
                    and board[x][y + 3] == color:
                return True

    # check \ diagonal spaces
    for x in range(3, NUM_COLUMNS):
        for y in range(NUM_ROWS - 3):
            if board[x][y] == color \
                    and board[x - 1][y + 1] == color \
                    and board[x - 2][y + 2] == color \
                    and board[x - 3][y + 3] == color:
                return True

    # check / diagonal spaces
    for x in range(0, NUM_COLUMNS - 3):
        for y in range(NUM_ROWS - 3):
            if board[x][y] == color \
                    and board[x + 1][y + 1] == color \
                    and board[x + 2][y + 2] == color \
                    and board[x + 3][y + 3] == color:
                return True

    return False


def draw(board):
    def show(x):
        if x is None:
            return '_'
        else:
            return str(x)

    rows = []
    for row_idx in reversed(range(NUM_ROWS)):
        rows.append('\t'.join([show(board[col_idx][row_idx]) for col_idx in range(NUM_COLUMNS)]))

    return '\n'.join(rows)
