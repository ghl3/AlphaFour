import json

import argparse

import ConnectFour as cf
import ai


def main():

    parser = argparse.ArgumentParser(description='General Tools')

    subparsers = parser.add_subparsers(help='Sub options')

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

    args = parser.parse_args()
    args.func(args)


def simulate(args):
    red_model = ai.load_model(args.red_model)
    yellow_model = ai.load_model(args.yellow_model)

    with open('{}metadata.txt'.format(args.output_prefix), 'w+') as f:
        f.write("Red Model:{}\nYellow Model: {}\n".format(args.red_model, args.yellow_model))

    for i in range(args.num_games):

        if i % 1000 == 0:
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

    print "All Games Run"


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
