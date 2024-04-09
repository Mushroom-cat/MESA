import sys
import copy
import torch
import random
import pickle
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pandas as pd

import scipy.sparse as sp

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, SEED, user_dict, existing_split):

    def sample():

        user = np.random.randint(1, existing_split * (usernum + 1))
        while len(user_train[user]) <= 1: user = np.random.randint(1, existing_split * (usernum + 1))

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break


        #user_attr = user_dict[user].squeeze(0)

        #return (user_attr, seq, pos, neg)
        return (user, seq, pos, neg)

    np.random.seed(SEED)

    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample())

    return zip(*one_batch)


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, user_dict, batch_size=64, maxlen=10, n_workers=1, existing_split=0.7):

        #sample_function(User, usernum, itemnum, batch_size, maxlen, np.random.randint(2e9), user_dict, existing_split)
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.user_dict = user_dict
        self.existing_split = existing_split


    def next_batch(self):
        return sample_function(self.User, self.usernum, self.itemnum, self.batch_size, self.maxlen, np.random.randint(2e9), self.user_dict, self.existing_split)

    # def close(self):
    #     for p in self.processors:
    #         p.terminate()
    #         p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_rating = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1

    score_data = pd.read_csv(
        'data/dbook/d_ratings.csv', names=['user_id', 'movie_id', 'rating', 'timestamp'],
        sep=",", engine='python'
    )

    for i in range(score_data.shape[0]):
        u = score_data.user_id[i]
        i = score_data.movie_id[i]
        r = score_data.rating[i]
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        User_rating[u].append(r)

    rows_ = []
    cols_ = []
    values_ = []

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            rows_ += ([user,] * len(User[user]))
            cols_ += (User[user])
            values_ += (User_rating[user])
        elif nfeedback > 50:
            user_train[user] = User[user][-50:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            rows_ += ([user, ] * len(User[user][-20:-2]))
            cols_ += (User[user][-20:-2])
            values_ += (User_rating[user][-20:-2])
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            rows_ += ([user, ] * len(User[user][:-2]))
            cols_ += (User[user][:-2])
            values_ += (User_rating[user][:-2])

    inter_mat = sp.coo_matrix((values_, (rows_, cols_)), shape=[usernum+1, itemnum+1])

    return [user_train, user_valid, user_test, usernum, itemnum, inter_mat]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, user_dict, args):
    [train, valid, test, usernum, itemnum, inter_mat] = copy.deepcopy(dataset)

    NDCG1 = 0.0
    HT1 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    #     users = range(1, usernum + 1)
    existing_split = args.existing_split
    if (1-existing_split) * (usernum + 1) > 10000:
        users = random.sample(range(int(existing_split * (usernum + 1)), usernum + 1), 10000)
    else:
        users = range(int(existing_split * (usernum + 1)), usernum + 1)

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(torch.tensor(u).to(args.device), np.array(seq), np.array(item_idx))

        #predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 1:
            NDCG1 += 1 / np.log2(rank + 2)
            HT1 += 1
        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1
        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG1 / valid_user, HT1 / valid_user, NDCG5 / valid_user, HT5 / valid_user, NDCG10 / valid_user, HT10 / valid_user


# # evaluate on val set
# def evaluate_valid(model, dataset, user_dict, args):
#     [train, valid, test, usernum, itemnum, inter_mat] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#
#     # if usernum>10000:
#     #     users = random.sample(range(1, usernum + 1), 10000)
#     # else:
#     #     users = range(1, usernum + 1)
#     existing_split = args.existing_split
#     if (1-existing_split) * (usernum + 1) > 10000:
#         users = random.sample(range(int(existing_split * (usernum + 1)), usernum + 1), 10000)
#     else:
#         users = range(int(existing_split * (usernum + 1)), usernum + 1)
#
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#
#         user_attr = user_dict[u].squeeze(0)
#
#         predictions = -model.predict(user_attr.to(args.device), np.array(seq), np.array(item_idx))
#
#         #predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0].item()
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user