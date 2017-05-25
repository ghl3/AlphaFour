
from random import randint


import ConnectFour as cf

def random_game():

    board = cf.create_board()

    current = cf.RED

    while True:

        current = cf.YELLOW if current == cf.RED else cf.RED

        print current
        print cf.draw(board)

        if cf.is_winner(board, cf.RED):
            print "WINNER: RED", cf.is_winner(board, cf.RED)[1]
            break
        elif cf.is_winner(board, cf.YELLOW):
            print "WINNER: YELLOW", cf.is_winner(board, cf.YELLOW)[1]
            break

        while True:
            to_play = randint(0, cf.NUM_COLUMNS-1)

            if cf.can_play(board, to_play):
                break

        cf.play(board, to_play, current)

if __name__ == '__main__':
    random_game()

