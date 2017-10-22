import os
import json
import glob

import argparse

import ConnectFour as cf
import ai


def parse_boolean(x):
    if x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        raise ValueError("Unknown boolean: {}".format(x))


def main():
    parser = argparse.ArgumentParser(description='General Tools')

    subparsers = parser.add_subparsers(help='Sub options')

    # Parser to play a game against the computer
    play_parser = subparsers.add_parser('play', help='Play a game against an AI')

    play_parser.add_argument('--ai', type=str, default='random',
                             help='The AI model to play against')

    play_parser.add_argument('--player-first', type=parse_boolean, default=True,
                             help='Whether the player goes first (RED is always first)')

    play_parser.set_defaults(func=play)

    # Parser to simulate a game
    sim_parser = subparsers.add_parser('simulate', help='Simulate a game')

    sim_parser.add_argument('--red-model', type=str, default='random',
                            help='The model to be used by one of the two players')

    sim_parser.add_argument('--yellow-model', type=str, default='random',
                            help='The model to be used by one of the two players')

    sim_parser.add_argument('--num-games', type=int, default=1000)

    sim_parser.add_argument('--output-prefix', type=str, required=True)

    sim_parser.set_defaults(func=simulate)

    # Parser to visualize data
    vis_parser = subparsers.add_parser('visualize', help='Visualize a saved game')

    vis_parser.add_argument('file', type=str, help='Game file to visualize')

    vis_parser.add_argument('--include-features', action='store_true',
                            help='Include features that are shown to the network')

    vis_parser.set_defaults(func=visualize)

    # Parser to generate training data
    play_parser = subparsers.add_parser('process', help='Process a list of games into training data for a model')

    play_parser.add_argument('games', nargs='+',
                             help='The games used to generate the training data')

    play_parser.add_argument('--output-prefix', required=True, type=str,
                             help='The output prefix for the generated features, targets files')

    play_parser.set_defaults(func=process)

    # Parse and determine what to do
    args = parser.parse_args()
    args.func(args)


def play(args):
    if args.player_first:
        player = cf.RED
        red_model = ai.PlayerModel()
        yellow_model = ai.load_model(args.ai)
    else:
        player = cf.YELLOW
        yellow_model = ai.PlayerModel()
        red_model = ai.load_model(args.ai)

    board = cf.create_board()

    # Alternate between Red/Yellow

    current_player = cf.RED
    current_model = red_model

    while True:

        move = current_model.get_move(board, current_player)
        board = cf.play(board, move, current_player)

        if cf.is_winner(board, cf.RED):
            print cf.draw(board)
            print "RED ({}) WINS".format('PLAYER' if player == cf.RED else 'COMPUTER')
            return
        elif cf.is_winner(board, cf.YELLOW):
            print cf.draw(board)
            print "YELLOW ({}) WINS".format('PLAYER' if player == cf.YELLOW else 'COMPUTER')
            return
        elif cf.is_tie(board):
            print cf.draw(board)
            print "TIE GAME"
            return
        else:
            current_player = cf.YELLOW if current_player == cf.RED else cf.RED
            current_model = yellow_model if current_player == cf.YELLOW else red_model

            # yellow_move = yellow_model.get_move(board, cf.YELLOW)
            # board = cf.play(board, yellow_move, cf.YELLOW)

            # if cf.is_winner(board, cf.RED):
            #    print cf.draw(board)
            #    print "YELLOW ({}) WINS".format('PLAYER' if player == cf.YELLOW else 'COMPUTER')
            #    return


def simulate(args):
    red_model = ai.load_model(args.red_model)
    yellow_model = ai.load_model(args.yellow_model)

    prefix_base_path = make_base_dir(args.output_prefix)

    # prefix_base_path = os.path.dirname(args.output_prefix)
    # os.makedirs(prefix_base_path)

    with open('{}metadata.txt'.format(args.output_prefix), 'w+') as f:
        f.write("Red Model:{}\nYellow Model: {}\n".format(args.red_model, args.yellow_model))

    for i in range(args.num_games):

        if i % 1000 == 0 and i > 0:
            print "Generating Game: {}".format(i)

        results = ai.play_game(red_model, yellow_model)

        data = results

        output_file = "{}/game_{}.json".format(args.output_prefix, i)
        with open(output_file, 'w+') as f:
            f.write(json.dumps(data))
            f.write('\n')

    print "All Games Run.  Saved to: {}".format(prefix_base_path)


def visualize(args):
    with open(args.file) as f:
        data = json.loads(f.read())

    for idx, turn in enumerate(data['turns']):
        print "Turn: {} {}".format(idx, turn['player'])
        print cf.draw(turn['board'])
        print "Move: {}->{}".format(turn['player'], turn['move'])
        if args.include_features:
            print ai.get_features_from_turn(turn['player'], turn['board'])
            print ai.get_target_from_turn(turn['player'], data['winner'])

        print '\n'

    print "Winner:", data['winner']
    print cf.draw(data['final'])


def get_features_from_game(game_data):
    # The current player's disks ar 1
    # opponent player's disks are 0

    game_features = []
    game_targets = []
    for turn in game_data['turns']:
        # Get the board's state AFTER the player makes the move
        # The target is "Will this player win IF they make this move"
        board = turn['board']
        board_after_move = cf.play(board, turn['move'], turn['player'])

        features = ai.get_features_from_turn(turn['player'], board_after_move)
        target = ai.get_target_from_turn(turn['player'], game_data['winner'])
        game_features.append(features)
        game_targets.append(target)

    return game_features, game_targets


def process(args):
    game_idx = 0

    prefix_base_path = make_base_dir(args.output_prefix)
    with open('{}features.csv'.format(args.output_prefix), 'w+') as feat_file:
        with open('{}targets.csv'.format(args.output_prefix), 'w+') as targ_file:
            for game_glob in args.games:
                for file in glob.glob(game_glob):

                    game_idx += 1
                    if game_idx % 1000 == 0 and game_idx > 0:
                        print "Processing Game: {}".format(game_idx)

                    with open(file) as f:
                        game_data = json.loads(f.read())

                        # We add the game index as the 0th feature
                        # so we can do fully out-of-sample validation
                        game_features, game_targets = get_features_from_game(
                            game_data)  # ame_data['turns'], game_data['winner'])
                        for turn_idx, row in enumerate(game_features):
                            feat_file.write(','.join([str(game_idx), str(turn_idx)] + [str(i) for i in row]))
                            feat_file.write('\n')

                        for turn_idx, row in enumerate(game_targets):
                            targ_file.write(','.join([str(game_idx), str(turn_idx)] + [str(i) for i in row]))
                            targ_file.write('\n')

                            #                        all_features.extend([[idx] + fs for fs in game_features])
                            #                        all_targets.extend([[idx] + ts for ts in game_targets])

                            # Features are saved as a CSV


# for row in all_features:
#            f.write(','.join([str(i) for i in row]))
#            f.write('\n')


#        for row in all_targets:
#            f.write(','.join([str(i) for i in row]))
#            f.write('\n')


def make_base_dir(prefix):
    prefix_base_path = os.path.dirname(prefix)
    try:
        os.makedirs(prefix_base_path)
    except OSError:
        pass

    return prefix_base_path


if __name__ == '__main__':
    main()
