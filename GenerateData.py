
from random import randint

import ConnectFour as cf

import argparse


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num-games', type=int, default=1000)
    parser.add_argument('--output-prefix', type=str, required=True)

    args = parser.parse_args()
    features, targets = generate_training_data(args.num_games)

    features_path = "{}features.csv".format(args.output_prefix)
    with open(features_path, 'w+') as f:
        for row in features:
            f.write(",".join([str(x) for x in row]))
            f.write('\n')

    targets_path = "{}targets.csv".format(args.output_prefix)
    with open(targets_path, 'w+') as f:
        for row in targets:
            f.write(",".join([str(x) for x in row]))
            f.write('\n')


def random_game():
    """
    Runs a full random game
    Returns a list of (board, next-mode) pairs winner:

    ([(player, board), (player, board), ...], winner)
    """

    boards = []

    board = cf.create_board()

    current = cf.RED

    while True:

        current = cf.YELLOW if current == cf.RED else cf.RED

        boards.append((current, cf.clone(board)))

        if cf.is_tie(board):
            return boards, None

        elif cf.is_winner(board, cf.RED):
            return boards, cf.RED

        elif cf.is_winner(board, cf.YELLOW):
            return boards, cf.YELLOW

        while True:
            to_play = randint(0, cf.NUM_COLUMNS-1)

            if cf.can_play(board, to_play):
                break

        cf.play(board, to_play, current)


def get_features_from_game(player_board_pairs, winner):

    # The current player's disks ar 1
    # opponent player's disks are 0

    game_features = []
    game_targets = []
    for current, board in player_board_pairs:
        features = cf.get_features_from_turn(current, board)
        target = cf.get_target_from_turn(current, board, winner)
        game_features.append(features)
        game_targets.append(target)

    return game_features, game_targets


def generate_training_data(num_games=1000):

    all_features = []
    all_targets = []

    for i in range(num_games):

        if i % 1000 == 0:
            print "Generating Game: {}".format(i)

        plays, winner = random_game()

        game_features, game_targets = get_features_from_game(plays, winner)
        all_features.extend(game_features)
        all_targets.extend(game_targets)

    return all_features, all_targets


if __name__ == '__main__':
    main()
