
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

    features_path = "{}features.txt".format(args.output_prefix)
    with open(features_path, 'w+') as f:
        for row in features:
            f.write(",".join([str(x) for x in row]))
            f.write('\n')

    targets_path = "{}targets.txt".format(args.output_prefix)
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

        if cf.is_winner(board, cf.RED):
            return boards, cf.RED

        elif cf.is_winner(board, cf.YELLOW):
            return boards, cf.YELLOW

        while True:
            to_play = randint(0, cf.NUM_COLUMNS-1)

            if cf.can_play(board, to_play):
                break

        try:
            cf.play(board, to_play, current)
        except Exception as e:
            return boards, None


def _to_int(x, current_player):
    if x == current_player:
        return 1
    elif x is None:
        return 0
    else:
        return -1


def get_features_from_turn(current_player, board, winner):
    features = [_to_int(x, current_player) for col in board for x in col]
    if winner==current_player:
        target = [1, 0, 0]
    elif winner is None:
        target = [0, 0, 1]
    else:
        target = [0, 1, 0]
    #target = [[1, 0, 0] if _to_int(winner, current_player)
    return features, target


def get_features_from_game(player_board_pairs, winner):

    # The current player's disks ar 1
    # opponent player's disks are 0

    game_features = []
    game_targets = []
    for current, board in player_board_pairs:
        features, target = get_features_from_turn(current, board, winner)
        game_features.append(features)
        game_targets.append(target)

    return game_features, game_targets


def generate_training_data(num_games=1000):

    all_features = []
    all_targets = []

    for _ in range(num_games):
        plays, winner = random_game()

        game_features, game_targets = get_features_from_game(plays, winner)
        all_features.extend(game_features)
        all_targets.extend(game_targets)

    return all_features, all_targets


if __name__ == '__main__':
    main()
