
import argparse
from abc import ABCMeta, abstractmethod

import random
import json

from random import randint

import ConnectFour as cf
import ai

import argparse


def main():

    parser = argparse.ArgumentParser(description='Simulate a game between two ConnectFour models')

    parser.add_argument('--red-model', type=str, default='random',
                        help='The model to be used by one of the two players')

    parser.add_argument('--yellow-model', type=str, default='random',
                        help='The model to be used by one of the two players')

    parser.add_argument('--num-games', type=int, default=1000)

    parser.add_argument('--output-prefix', type=str, required=True)

    args = parser.parse_args()

    red_model = ai.load_model(args.red_model)
    yellow_model = ai.load_model(args.yellow_model)

    with open('{}metadata.txt'.format(args.output_prefix), 'w+') as f:
        f.write("Red Model:{}\nYellow Model: {}\n".format(args.red_model, args.yellow_model))

    for i in range(args.num_games):

        if i % 1000 == 0:
            print "Generating Game: {}".format(i)

        boards, winner = ai.play_game(red_model, yellow_model)

        data = {'boards': boards,
                'winner': winner}

        output_file = "{}/game_{}.json".format(args.output_prefix, i)
        with open(output_file, 'w+') as f:
            f.write(json.dumps(data))
            f.write('\n')

        #game_features, game_targets = get_features_from_game(plays, winner)
        #all_features.extend(game_features)
        #all_targets.extend(game_targets)

    print "All Games Run"

    # return all_features, all_targets

    # features, targets = generate_training_data(args.num_games)

    # features_path = "{}features.csv".format(args.output_prefix)
    # with open(features_path, 'w+') as f:
    #     for row in features:
    #         f.write(",".join([str(x) for x in row]))
    #         f.write('\n')

    # targets_path = "{}targets.csv".format(args.output_prefix)
    # with open(targets_path, 'w+') as f:
    #     for row in targets:
    #         f.write(",".join([str(x) for x in row]))
    #         f.write('\n')

#
# def simulate_game(red, yellow):
#     """
#     Runs a full random game
#     Returns a list of (board, next-mode) pairs winner:
#
#     ([(player, board), (player, board), ...], winner)
#     """
#
#     boards = []
#
#     board = cf.create_board()
#
#     current = random.choice([cf.RED, cf.YELLOW])
#
#     while True:
#
#         # Pick the next player to go
#         current = cf.YELLOW if current == cf.RED else cf.RED
#
#         # Save the board that this player faced
#         boards.append((current, cf.clone(board)))
#
#         # Determine their play
#         current_model = red if current==cf.RED else yellow
#         move = current_model.move(board)
#
#         # Update the board
#         cf.play(board, move, current)
#
#         # Check the results
#         if cf.is_tie(board):
#             return boards, None
#
#         elif cf.is_winner(board, cf.RED):
#             return boards, cf.RED
#
#         elif cf.is_winner(board, cf.YELLOW):
#             return boards, cf.YELLOW
#
#         else:
#             pass
#        while True:
#            to_play = randint(0, cf.NUM_COLUMNS-1)
#
#            if cf.can_play(board, to_play):
#                break



if __name__ == '__main__':
    main()
