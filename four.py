import os
import json
import glob

import argparse

import ConnectFour as cf
import ai


def main():
    parser = argparse.ArgumentParser(description='General Tools')

    subparsers = parser.add_subparsers(help='Sub options')

    # Parser to play a game against the computer
    play_parser = subparsers.add_parser('play', help='Play a game against an AI')

    play_parser.add_argument('--ai', type=str, default='random',
                             help='The AI model to play against')

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
    player = cf.RED
    computer = cf.YELLOW

    model = ai.load_model(args.ai)

    board = cf.create_board()

    while True:

        print cf.draw(board)
        print "\t".join(map(str, range(0, 7)))
        player_move = input('Next move (Enter 0-7): \n')

        board = cf.play(board, player_move, player)

        if cf.is_winner(board, player):
            print cf.draw(board)
            print "PLAYER WINS"
            return

        computer_move = model.get_move(board, computer)
        board = cf.play(board, computer_move, computer)
        if cf.is_winner(board, computer):
            print cf.draw(board)
            print "COMPUTER WINS"
            return


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

            # game_features, game_targets = get_features_from_game(plays, winner)
            # all_features.extend(game_features)
            # all_targets.extend(game_targets)

    print "All Games Run.  Saved to: {}".format(prefix_base_path)


def visualize(args):
    with open(args.file) as f:
        data = json.loads(f.read())

    for idx, turn in enumerate(data['turns']):
        print "Turn: {} {}".format(idx, turn['player'])
        print cf.draw(turn['board'])
        print "Move: {}->{}".format(turn['player'], turn['move'])
        print '\n'

    print "Winner:", data['winner']
    print cf.draw(data['final'])


def get_features_from_game(turns, winner):
    # The current player's disks ar 1
    # opponent player's disks are 0

    game_features = []
    game_targets = []
    for turn in turns:
        features = ai.get_features_from_turn(turn['player'], turn['board'])
        target = ai.get_target_from_turn(turn['player'], turn['board'], winner)
        game_features.append(features)
        game_targets.append(target)

    return game_features, game_targets


def process(args):
    all_features = []
    all_targets = []

    idx = 0

    for game_glob in args.games:
        for file in glob.glob(game_glob):

            idx += 1
            if idx % 1000 == 0 and idx > 0:
                print "Processing Game: {}".format(idx)

            with open(file) as f:
                game_data = json.loads(f.read())

                # We add the game index as the 0th feature
                # so we can do fully out-of-sample validation
                game_features, game_targets = get_features_from_game(game_data['turns'], game_data['winner'])
                all_features.extend([[idx] + fs for fs in game_features])
                all_targets.extend([[idx] + ts for ts in game_targets])

    # Features are saved as a CSV

    prefix_base_path = make_base_dir(args.output_prefix)

    with open('{}features.csv'.format(args.output_prefix), 'w+') as f:
        for row in all_features:
            f.write(','.join([str(i) for i in row]))
            f.write('\n')

    with open('{}targets.csv'.format(args.output_prefix), 'w+') as f:
        for row in all_targets:
            f.write(','.join([str(i) for i in row]))
            f.write('\n')


def make_base_dir(prefix):
    prefix_base_path = os.path.dirname(prefix)
    try:
        os.makedirs(prefix_base_path)
    except OSError:
        pass

    return prefix_base_path


if __name__ == '__main__':
    main()
