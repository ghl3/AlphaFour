
import pandas as pd
import random
import math

def load_data(prefix):
    features = pd.read_csv('{}/features.csv'.format(prefix),
                           names=['game_idx', 'turn_idx'] + range(42),
                           dtype='int')

    targets = pd.read_csv('{}/targets.csv'.format(prefix),
                          names=['game_idx', 'turn_idx', 'win', 'lose', 'draw'],
                          dtype='int')

    # Split into training and holdout,
    # ensuring separate games
    train_games = pd.Series(features.game_idx.unique()).sample(frac=0.8).values #games = #features.game_idx.unique()

    features_train = features[features.game_idx.isin(train_games)].drop(['game_idx', 'turn_idx'], axis=1)
    targets_train = targets[targets.game_idx.isin(train_games)].drop(['game_idx', 'turn_idx'], axis=1)

    tidx = list(features_train.index)
    random.shuffle(tidx)
    features_train = features_train.loc[tidx]
    targets_train = targets_train.loc[tidx]

    features_test = features[~features.game_idx.isin(train_games)].drop(['game_idx', 'turn_idx'], axis=1)
    targets_test = targets[~targets.game_idx.isin(train_games)].drop(['game_idx', 'turn_idx'], axis=1)

    return features_train, targets_train, features_test, targets_test




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


