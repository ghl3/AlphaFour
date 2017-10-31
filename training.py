
import pandas as pd
import random
import math
import time

import tensorflow as tf
from keras import backend as K


class Dataset(object):

    def __init__(self,
                 X_train, y_train,
                 X_test, y_test):

        assert X_train.index.equals(y_train.index)
        assert X_test.index.equals(y_test.index)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


class DataLoader(object):

    def __init__(self, prefix='training_data', frac_train=0.80, frac_test=0.20):

        assert frac_train + frac_test == 1.00

        self.frac_train = frac_train
        self.frac_test = frac_test
        self._prefix = prefix
        self._datasets = []
        self._sizes = []

    def add_dataset(self, name, num_games=None):
        self._datasets.append(name)
        self._sizes.append(num_games)
        return self

    def load(self):

        X_trains = []
        y_trains = []
        X_tests = []
        y_tests = []

        for name, count in zip(self._datasets, self._sizes):
            features = pd.read_csv('{}/{}/features.csv'.format(self._prefix, name),
                                   names=['game_idx', 'turn_idx'] + range(42),
                                   dtype='int')

            targets = pd.read_csv('{}/{}/targets.csv'.format(self._prefix, name),
                                  names=['game_idx', 'turn_idx', 'win', 'lose', 'draw'],
                                  dtype='int')

            assert len(features) == len(targets), "Num Features {} doesn't match Num Targets {} for {}".format(len(features), len(targets), name)

            # Split into training and holdout,
            # ensuring separate games

            if count is None:
                games = pd.Series(features.game_idx.unique())
            else:
                games = pd.Series(features.game_idx.unique()).sample(n=count)
            train_games = games.sample(frac=self.frac_train)
            test_games = games.loc[~games.index.isin(train_games.index)]

            features_train = features[features.game_idx.isin(train_games)].assign(name=name).set_index(['name', 'game_idx', 'turn_idx'])
            targets_train = targets[targets.game_idx.isin(train_games)].drop(['game_idx', 'turn_idx'], axis=1).set_index(features_train.index)

            features_test = features[features.game_idx.isin(test_games)].assign(name=name).set_index(['name', 'game_idx', 'turn_idx'])
            targets_test = targets[targets.game_idx.isin(test_games)].drop(['game_idx', 'turn_idx'], axis=1).set_index(features_test.index)

            X_trains.append(features_train)
            y_trains.append(targets_train)
            X_tests.append(features_test)
            y_tests.append(targets_test)


        X_train = pd.concat(X_trains).sample(frac=1)
        y_train = pd.concat(y_trains).loc[X_train.index]
        X_test = pd.concat(X_tests).sample(frac=1)
        y_test = pd.concat(y_tests).loc[X_test.index]

        return Dataset(
            X_train,
            y_train,
            X_test,
            y_test)


def get_batch_iter(batch_size, batch_idx, dfs):

    length = len(dfs[0])
    for df in dfs:
        assert len(df) == length

    batches_per_df = int(math.ceil(length / batch_size))

    local_idx = batch_idx % batches_per_df

    start = local_idx*batch_size
    end = (local_idx+1)*batch_size

    return [df.iloc[start:end] for df in dfs]


def get_batch_random(batch_size, _, dfs):
    mask = pd.Series(dfs[0].index).sample(n=batch_size, replace=False)
    return [df.loc[mask] for df in dfs]


def get_batch(batch_size, batch_idx, dfs, how='iter'):
    if how == 'iter':
        return get_batch_iter(batch_size, batch_idx, dfs)
    elif how == 'random':
        return get_batch_random(batch_size, batch_idx, dfs)
    else:
        raise Exception()


def get_batch_iter(batch_size, batch_idx, dfs):

    length = len(dfs[0])
    for df in dfs:
        assert len(df) == length

    batches_per_df = int(math.ceil(length / batch_size))

    local_idx = batch_idx % batches_per_df

    start = local_idx*batch_size
    end = (local_idx+1)*batch_size

    return [df.iloc[start:end] for df in dfs]


def get_batch_random(batch_size, _, dfs):
    mask = pd.Series(dfs[0].index).sample(n=batch_size, replace=False)
    return [df.loc[mask] for df in dfs]


def get_batch(batch_size, batch_idx, dfs, how='iter'):
    if how == 'iter':
        return get_batch_iter(batch_size, batch_idx, dfs)
    elif how == 'random':
        return get_batch_random(batch_size, batch_idx, dfs)
    else:
        raise Exception()


def train(graph, output_prefix, dataset,
          batch_size, num_batches, batch_how='iter'):

    # To be run, a graph must have the following:
    # - Tensor: board:0
    # - Tensor: outcome:0
    # - Tensor: Merge/MergeSummary:0
    # - Operation: init_op
    # - Operation: training/train_step

    board = graph.get_tensor_by_name('board:0')
    outcome = graph.get_tensor_by_name('outcome:0')
    all_summaries = graph.get_tensor_by_name('Merge/MergeSummary:0')

    loss = graph.get_tensor_by_name('evaluation/loss:0')
    accuracy = graph.get_tensor_by_name('evaluation/accuracy:0')

    init_op = graph.get_operation_by_name('init')
    train_step = graph.get_operation_by_name('training/train_step')

    with tf.Session(graph=graph) as sess:

        K.set_session(sess)
        sess.run(init_op)

        train_writer = tf.summary.FileWriter(output_prefix, sess.graph)

        print "Running {}".format(output_prefix)

        t = time.time()
        delta_t = 0

        for i in range(num_batches):

            if i % 1000 == 0:

                delta_t = time.time() - t
                t = time.time()

                summary = sess.run(all_summaries, feed_dict={board: dataset.X_test, outcome: dataset.y_test})
                train_writer.add_summary(summary, i)

                acc_val = accuracy.eval(feed_dict={board: dataset.X_test, outcome: dataset.y_test})
                loss_val = loss.eval(feed_dict={board: dataset.X_test, outcome: dataset.y_test})

                print "Batch {:8} Hold-Out Accuracy: {:.4f} ({:d}/{:d}) Loss: {:.4f} Time taken: {:.1f}s".format(i, acc_val.mean(), int(acc_val.sum()), int(acc_val.size),
                                                                                                             loss_val.mean(), delta_t)

            batch = get_batch(batch_size, i, [dataset.X_train, dataset.y_train], how=batch_how)
            train_step.run(feed_dict={board: batch[0], outcome: batch[1]})

        print "\nDONE TRAINING"
        print "FINAL ACCURACY: {:.4f} FINAL LOSS: {:.4f}".format(acc_val.mean(), loss_val.mean())
        train_writer.close()

        model_dir = '{}/model'.format(output_prefix)
        print "SAVING TO: {}".format(model_dir)
        tf.train.Saver().save(sess, model_dir)

