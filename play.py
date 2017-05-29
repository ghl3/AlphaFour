
from collections import OrderedDict

import ConnectFour as cf

from random import randint

import numpy as np
import tensorflow as tf


class Model(object):

    def get_move(self, board, player):
        raise NotImplementedException()


class RandomModel(Model):

    def get_move(self, board, player):
        computer_move = randint(0, cf.NUM_COLUMNS-1)
        return computer_move


class AI(Model):

    def __init__(self, model_path):
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default() as g:
            saver = tf.train.import_meta_graph('./models/{}/model.meta'.format(model_path))
            saver.restore(sess, tf.train.latest_checkpoint('./models/{}/'.format(model_path)))
            prediction = graph.get_operation_by_name("predictions/preds/Softmax").values()[0]
            board = graph.get_operation_by_name("board").values()[0]

        self.sess = sess
        self.board = board
        self.prediction = prediction

    def get_probabilities(self, row):
        return self.sess.run(self.prediction, feed_dict={self.board: np.array(row).reshape(1, 42)})


    def get_move(self, board, player):

        possible_moves = OrderedDict()

        for i in range(0, cf.NUM_COLUMNS):

            if cf.can_play(board, i):

                b = cf.clone(board)
                cf.play(b, i, player)
                features = cf.get_features_from_turn(player, b)
                win_prob = self.get_probabilities(features)[0][0]
                possible_moves[i] = win_prob


        print "Possible Moves", possible_moves
        return AI.get_best_move(possible_moves)

    @staticmethod
    def get_best_move(possible_moves):

        max_prob = None
        best_move = None

        for move, prob in possible_moves.iteritems():
            if max_prob:
                if prob > max_prob:
                    max_prob = prob
                    best_move = move
            else:
                max_prob = prob
                best_move = move

        return best_move


def play(player_first, model):

    player = cf.RED
    computer = cf.YELLOW

    board = cf.create_board()
    print cf.draw(board)

    while True:

        player_move = input('Next move (Enter 0-7): \n')
        cf.play(board, player_move, player)
        print 'PLAYER MOVE: \n'
        print cf.draw(board)
        if cf.is_winner(board, player):
            print "PLAYER WINS"
            return

        computer_move = model.get_move(board, computer)
        #computer_move = randint(0, cf.NUM_COLUMNS-1)
        cf.play(board, computer_move, computer)
        print 'COMPUTER MOVE: \n'
        print cf.draw(board)
        if cf.is_winner(board, computer):
            print "COMPUTER WINS"
            return


if __name__ == '__main__':
    play(True, AI('cov2d_2017_05_29_14:37:42'))


