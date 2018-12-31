from abc import abstractmethod, ABCMeta
import random

import numpy as np
import ConnectFour as cf


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_move(self, board, player):
        """
        Return an integer that represents the move
        """
        raise NotImplementedError()


class PlayerModel(Model):
    def get_move(self, board, player):
        print cf.draw(board)
        print "\t".join(map(str, range(0, 7)))
        player_move = input('Next move (Enter 0-7): \n')

        assert player_move in range(0, 7)
        return player_move


class RandomModel(Model):
    def get_move(self, board, player):
        while True:
            to_play = random.randint(0, cf.NUM_COLUMNS - 1)
            if cf.can_play(board, to_play):
                return to_play


class AI(Model):
    """
    A model that wraps a Tensorflow Model for AI
    """

    def __init__(self, model_path, greedy=False, session_config=None):

        # Don't import tensorflow at the top so that the client can control
        # when it's imported.  This may help avoid deadlock issues...
        import tensorflow as tf

        graph = tf.Graph()
        sess = tf.Session(graph=graph, config=session_config)
        with graph.as_default() as g:
            saver = tf.train.import_meta_graph(model_path + "/model.meta")
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            prediction = g.get_operation_by_name("preds/Softmax").values()[0]
            board = g.get_operation_by_name("board").values()[0]

        self.greedy = greedy
        self.sess = sess
        self.board = board
        self.prediction = prediction

    def get_probabilities(self, row):
        return self.sess.run(self.prediction, feed_dict={self.board: np.array(row).reshape(1, 42)})

    def get_move(self, board, player):

        possible_moves = np.zeros(cf.NUM_COLUMNS)

        for i in range(0, cf.NUM_COLUMNS):

            if cf.can_play(board, i):
                features = get_features_from_turn(player, cf.play(board, i, player))
                win_prob = self.get_probabilities(features)[0][0]
                possible_moves[i] = win_prob

        return self.get_best_move(possible_moves)

    @staticmethod
    def softmax(probs, theta):
        ps = np.exp(probs * theta)
        ps /= np.sum(ps)
        return ps

    def get_best_move(self, possible_moves):
        # Use an aggressive softmax to turn the
        # probability of winning GIVEN a move into
        # a probability of doing each move.
        # We could be fully greedy here instead
        # but we add randomness for exploration
        move_probability = AI.softmax(possible_moves, theta=25)
        if self.greedy:
            return np.argmax(move_probability)
        else:
            return np.random.choice(range(0, cf.NUM_COLUMNS), p=move_probability)


def load_model(model_name, *args, **kwargs):
    if model_name == 'random':
        return RandomModel()
    else:
        return AI(model_name, *args, **kwargs)


def play_game(red, yellow):
    """
    Runs a game of ConnectFour using the given models
    Returns a list of (board, next-mode) pairs winner:

    ([(player, board), (player, board), ...], winner)
    """

    turns = []

    board = cf.create_board()

    current_player = random.choice([cf.RED, cf.YELLOW])

    while True:

        # Pick the next player to go
        current_player = cf.YELLOW if current_player == cf.RED else cf.RED

        # Determine their play
        current_model = red if current_player == cf.RED else yellow
        move = current_model.get_move(board, current_player)

        # Save the board that this player faced
        turns.append({'player': current_player,
                      'board': board,
                      'move': move})

        # Update the board
        board = cf.play(board, move, current_player)

        # Check the results
        if cf.is_tie(board):
            return {'turns': turns,
                    'winner': None,
                    'final': board}

        elif cf.is_winner(board, cf.RED):
            return {'turns': turns,
                    'winner': cf.RED,
                    'final': board}

        elif cf.is_winner(board, cf.YELLOW):
            return {'turns': turns,
                    'winner': cf.YELLOW,
                    'final': board}

        else:
            pass


def get_features_from_turn(current_player, board):
    return [_to_int(x, current_player) for col in board for x in col]


def _to_int(x, current_player):
    if x == current_player:
        return 1
    elif x is None:
        return 0
    else:
        return -1


def get_target_from_turn(current_player, winner):
    if winner == current_player:
        return [1, 0, 0]
    elif winner is None:
        return [0, 0, 1]
    else:
        return [0, 1, 0]
