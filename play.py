
from collections import OrderedDict

import ConnectFour as cf

from random import randint

import numpy as np
import tensorflow as tf

import argparse



def play(player_first, model):

    player = cf.RED
    computer = cf.YELLOW

    board = cf.create_board()

    while True:

        print cf.draw(board)
        print "\t".join(map(str, range(0, 7)))
        player_move = input('Next move (Enter 0-7): \n')

        cf.play(board, player_move, player)
        if cf.is_winner(board, player):
            print cf.draw(board)
            print "PLAYER WINS"
            return

        computer_move = model.get_move(board, computer)
        cf.play(board, computer_move, computer)
        if cf.is_winner(board, computer):
            print cf.draw(board)
            print "COMPUTER WINS"
            return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--model', type=str, required=True,
                        help="Model to play again.  Options are ('random' or a name in the './models' directory")

    args = parser.parse_args()

    if args.model == 'random':
        model = RandomModel()
    else:
        model = AI(args.model)

    play(True, model)


