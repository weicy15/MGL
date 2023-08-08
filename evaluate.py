import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict

from models import Model
from util import metric

import load_data

import os
import time
import shutil
from tqdm import tqdm



def my_collate_test(batch):
    user_id = [item[0] for item in batch]
    pos_item_tensor = [item[1] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.stack(pos_item_tensor)

    return [user_id, pos_item]




cuda_device = '4'
device = torch.device("cuda:{0}".format(cuda_device))

loadFilename = "model.tar"
checkpoint = torch.load(loadFilename, map_location=device)
sd = checkpoint['sd']
opt = checkpoint['opt']

interact_train, interact_test, social, user_num, item_num, user_feature, item_feature = load_data.data_load(opt.dataset_name, social_data=opt.social_data, test_dataset= True, bottom=opt.implcit_bottom)
Data = load_data.Data(interact_train, interact_test, social, user_num, item_num, user_feature, item_feature)

print('Building dataloader >>>>>>>>>>>>>>>>>>>')
test_dataset = Data.test_dataset
test_loader = DataLoader(
    test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=my_collate_test)




model = Model(Data, opt, device)
model.load_state_dict(best_checkpoint['sd'])
model = model.to(device)
model.eval()
user_historical_mask = Data.user_historical_mask.to(device)

NDCG = defaultdict(list)
RECALL = defaultdict(list)
MRR = defaultdict(list)

head_NDCG = defaultdict(list)
head_RECALL = defaultdict(list)
tail_NDCG = defaultdict(list)
tail_RECALL = defaultdict(list)
body_NDCG = defaultdict(list)
body_RECALL = defaultdict(list)

with tqdm(total=len(test_loader), desc="predicting") as pbar:
    for i, (user_id, pos_item) in enumerate(test_loader):
        user_id = user_id.to(device)
        score = model.predict(user_id)
        score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
        ground_truth = pos_item.detach().numpy()

        for K in opt.K_list:
            ndcg, recall, mrr = metric.ranking_meansure_testset(score, ground_truth, K, list(Data.testSet_i.keys()))
            head_ndcg, head_recall, tail_ndcg, tail_recall, body_ndcg, body_recall = metric.ranking_meansure_degree_testset(score, ground_truth, K, Data.itemDegrees, opt.seperate_rate, list(Data.testSet_i.keys()))
            NDCG[K].append(ndcg)
            RECALL[K].append(recall)
            MRR[K].append(mrr)
        
            head_NDCG[K].append(head_ndcg)
            head_RECALL[K].append(head_recall)
            tail_NDCG[K].append(tail_ndcg)
            tail_RECALL[K].append(tail_recall)
            body_NDCG[K].append(body_ndcg)
            body_RECALL[K].append(body_recall)

        pbar.update(1)

print(opt)
print(model.name)
for K in opt.K_list:
    print("NDCG@{}: {}".format(K, np.mean(NDCG[K])))
    print("RECALL@{}: {}".format(K, np.mean(RECALL[K])))
    print("MRR@{}: {}".format(K, np.mean(MRR[K])))
    print('\r\r')
    print("head_NDCG@{}: {}".format(K, np.mean(head_NDCG[K])))
    print("head_RECALL@{}: {}".format(K, np.mean(head_RECALL[K])))
    print('\r\r')
    print("tail_NDCG@{}: {}".format(K, np.mean(tail_NDCG[K])))
    print("tail_RECALL@{}: {}".format(K, np.mean(tail_RECALL[K])))
    print('\r\r')
    print("body_NDCG@{}: {}".format(K, np.mean(body_NDCG[K])))
    print("body_RECALL@{}: {}".format(K, np.mean(body_RECALL[K])))
        

