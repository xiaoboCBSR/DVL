import argparse
import os
from os.path import join
import json
import pandas as pd
import time
import numpy as np
import torch


def str2bool(v):
    # convert string to boolean type for argparser input
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_or_none(v):
    # convert string to boolean type for argparser input
    if v is None:
        return None
    if v.lower() == 'none':
        return None
    else:
        return v


# helper functions for LDA arguments
def dic2name(dic):
    return '_'.join(["{}-{}".format(k, dic[k]) for k in sorted(dic)])


def name2dic(s):
    return {x.split('-')[0]: x.split('-')[1] for x in s.split('_')}


def get_valid_types(TYPENAME):
    """
    Get the label names
    """
    with open(join('configs', 'types.json'), 'r') as typefile:
        valid_types = json.load(typefile)[TYPENAME]
    return valid_types


def load_tmp_df(tmp_path, name):
    """
    load dataframe from pickle
    """
    start = time.time()
    pkl_file = join(tmp_path, "{}.pkl".format(name))
    if os.path.exists(pkl_file):
        print("{} pickle file found, loading...".format(pkl_file))
        df = pd.read_pickle(pkl_file)
    else:
        print("{} pickle file not found, please generate it first...".format(pkl_file))
    print("{} Load complete. Time {}".format(name, time.time() - start))
    return df


def logSumExpTensor(vec):
    # vec -> 16, tag_size
    batch_size = vec.size()[0]
    vec = vec.view(batch_size, -1)
    max_score = torch.max(vec, 1)[0]
    max_score_broadcast = max_score.view(-1, 1).expand(-1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))


def logNormalizeTensor(a):
    denom = logSumExpTensor(a)
    if len(a.size()) == 2:
        denom = denom.view(-1, 1).expand(-1, a.size()[1])
    elif len(a.size()) == 3:
        denom = denom.view(a.size()[0], 1, 1).expand(-1, a.size()[1], a.size()[2])
    return (a - denom)


def logNormalize(a):
    denom = np.logaddexp.reduce(a, 1)
    return (a.transpose() - denom).transpose()


def logDot(a, b):
    # numeric stable way of calculating log (e^a, e^b)
    max_a = np.amax(a)
    max_b = np.amax(b)

    C = np.dot(np.exp(a - max_a), np.exp(b - max_b))
    np.log(C, out=C)
    # else:
    #   np.log(C + 1e-300, out=C)

    C += max_a + max_b

    return C
