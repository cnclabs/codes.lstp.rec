import argparse
import numpy as np
import json
import math
import pandas as pd
import pickle
import random
import sys
import time, datetime
from utils import RecallPrecision_ATk, NDCGatK_r
import multiprocessing as mp
import statistics

#Align LightGCN's metrics
parser=argparse.ArgumentParser(description='top-N rec.')
parser.add_argument('--emb_file', type=str, help='emb_file')
parser.add_argument('--dataset', type=str, help='book, beauty')
parser.add_argument('--K', type=int, help='top-k')

args=parser.parse_args()
emb_file=args.emb_file
dataset=args.dataset
K=args.K

user_emb = {}
item_emb = {}

print('Loading the embeddings...')
with open(emb_file, 'r') as f:
    if emb_file.split('_')[1][:3] == 'lfm':
        for line in f:
            line=line.split()
            prefix=line.pop(0)
            if prefix.split('_')[0]=='user':
                user_emb.update({ prefix: np.array(line[2:], dtype=np.float32) })
            elif prefix.split('_')[0]=='item':
                item_emb.update({ prefix: np.array(line[2:], dtype=np.float32) })
            else:
                continue
    else:
        for line in f:
            line=line.split()
            prefix=line.pop(0)
            if prefix.split('_')[0]=='user':
                user_emb.update({ prefix: np.array(line, dtype=np.float32) })
            elif prefix.split('_')[0]=='item':
                item_emb.update({ prefix: np.array(line, dtype=np.float32) })
            else:
                continue


df=pd.read_csv('data/' + dataset + '_test.txt', delimiter='\t', names=['user', 'item', 'count'])
watched_training_log=pd.read_csv('data/' + dataset + '_train.txt', delimiter='\t', names=['user_id', 'item_id', 'count'])

common_user = list(set(df['user'].unique()) & set(watched_training_log['user_id'].unique()))
item_list = list(item_emb.keys())

def rank(user):
    ans = df[df['user'] == user]['item'].unique()
    #remove watched items
    watch = watched_training_log[watched_training_log['user_id'] == user]['item_id'].unique()
    unwatch = list(set(item_list) - set(watch))
    scores = np.zeros(len(unwatch), dtype=float)
    for i in range(len(unwatch)):
        scores[i] = np.dot(a=user_emb[user], b=item_emb[unwatch[i]])
    top_index = scores.argsort()[-K:][::-1]
    predict = np.array(unwatch)[top_index]
    hit = []
    for i in predict:
        if i in ans:
            hit.append(1)
        else:
            hit.append(0)
    #In utils, ans and hit are sent in batch(2-dimension list and np.array)
    ans = [ans]
    hit = np.array([hit])
    P = RecallPrecision_ATk(ans, hit, K)['precision']
    R = RecallPrecision_ATk(ans, hit, K)['recall']
    N = NDCGatK_r(ans, hit, K)
    return P, R, N

print('Evaluating...')
num_pool = mp.cpu_count()
pool = mp.Pool(num_pool)
precision_list, recall_list, ndcg_list = zip(*pool.map(rank, common_user))
pool.close()
pool.join()

print('==================')
print('File: ' + str(emb_file))
print('Evaluated users: ' + str(len(recall_list)))
print('precision@' + str(K) + ': ' + str(statistics.mean(precision_list)))
print('recall@' + str(K) + ': ' + str(statistics.mean(recall_list)))
print('NDCG@' + str(K) + ': ' + str(statistics.mean(ndcg_list)))
print('==================')
