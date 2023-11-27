import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict

from models import Model

import load_data

import os
import time
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
import pandas as pd
import metric



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='book_crossing')
    parser.add_argument("--social_data", type=bool, default=False)
    # test_set/cv/split
    parser.add_argument("--load_mode", type=str, default='test_set')

    parser.add_argument("--implcit_bottom", type=int, default=None)
    parser.add_argument("--cross_validate", type=int, default=None)
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--user_fre_threshold", type=int, default=None)
    parser.add_argument("--item_fre_threshold", type=int, default=None)

    parser.add_argument("--loadFilename", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=300)

    parser.add_argument("--embedding_size", type=int, default=8)
    parser.add_argument("--id_embedding_size", type=int, default=64)
    parser.add_argument("--dense_embedding_dim", type=int, default=16)

    parser.add_argument("--L", type=int, default=3)
    
    parser.add_argument("--link_topk", type=int, default=10)

    parser.add_argument("--reg_lambda", type=float, default=0.02)
    parser.add_argument("--top_rate", type=float, default=0.1)
    parser.add_argument("--convergence", type=float, default=40)
    parser.add_argument("--seperate_rate", type=float, default=0.2)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)


    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)


    parser.add_argument("--K_list", type=int, nargs='+', default=[10, 20, 50])

    opt = parser.parse_args()
    
    return opt

cuda_device = '4'


def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]
    neg_item = [item[2] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)
    neg_item = torch.LongTensor(neg_item)

    return [user_id, pos_item, neg_item]


def collate_test_i2i(batch):
    item1 = [item[0] for item in batch]
    item2 = [item[1] for item in batch]

    item1 = torch.LongTensor(item1)
    item2 = torch.LongTensor(item2)


    return [item1, item2]



def one_train(Data, opt):
    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    test_dataset = Data.test_dataset
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=my_collate_test)

    device = torch.device("cuda:{0}".format(cuda_device))

    print(device)
    index = [Data.interact_train['userid'].tolist(), Data.interact_train['itemid'].tolist()]
    value = [1.0] * len(Data.interact_train)

    interact_matrix = torch.sparse_coo_tensor(index, value, (Data.user_num, Data.item_num)).to(device)

    i2i = torch.sparse.mm(interact_matrix.t(), interact_matrix)

    def sparse_where(A):
        A = A.coalesce()
        A_values = A.values()
        A_indices = A.indices()
        A_values = torch.where(A_values > 1, A_values.new_ones(A_values.shape), A_values)
        return torch.sparse_coo_tensor(A_indices, A_values, A.shape).to(A.device)

    
    i2i = sparse_where(i2i)

    def get_0_1_array(item_num,rate=0.2):
        zeros_num = int(item_num * rate)
        new_array = np.ones(item_num * item_num)
        new_array[:zeros_num] = 0 
        np.random.shuffle(new_array)
        re_array = new_array.reshape(item_num * item_num)
        re_array = torch.from_numpy(re_array).to_sparse().to(device)
        return re_array


    mask = get_0_1_array(Data.item_num)
    i2i = mask * i2i * mask.t()

    i2i = i2i.coalesce()

    item1 = i2i.indices()[0].tolist()
    item2 = i2i.indices()[1].tolist()
    i2i_pair = list(zip(item1, item2))


    print("building model >>>>>>>>>>>>>>>")
    model = Model(Data, opt, device)


    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print('Start training...')
    start_epoch = 0
    directory = directory_name_generate(model, opt, "no early stop")
    model = model.to(device)

    support_loader = DataLoader(i2i_pair, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_test_i2i)

    for epoch in range(start_epoch, opt.epoch):
        model.train()

        train_loader = DataLoader(Data.train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)

        support_iter = iter(support_loader)

        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (user_id, pos_item, neg_item) in enumerate(train_loader):
                
                user_id = user_id.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                item1, item2 = next(support_iter)
                item1 = item1.to(device)
                item2 = item2.to(device)

                support_loss = model.i2i(item1, item2) + opt.reg_lambda * model.reg(item1)

                weight_for_local_update = list(model.generator.encoder.state_dict().values())

                grad = torch.autograd.grad(support_loss, model.generator.encoder.parameters(), create_graph=True, allow_unused=True)
                fast_weights = []
                for i, weight in enumerate(weight_for_local_update):
                    fast_weights.append(weight - opt.local_lr * grad[i])

                query_loss = model.q_forward(user_id, pos_item, neg_item, fast_weights)

                loss = query_loss + opt.beta * support_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)


    torch.save({
        'sd': model.state_dict(),
        'opt':opt,
    }, 'model.tar')

    # remove the historical interaction in the prediction
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
            # pos_item (uuu * item_num)
            # uuu * item_num
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
    

opt = get_config()
interact_train, interact_test, social, user_num, item_num, user_feature, item_feature = load_data.data_load(opt.dataset_name, social_data=opt.social_data, test_dataset= True, bottom=opt.implcit_bottom)
Data = load_data.Data(interact_train, interact_test, social, user_num, item_num, user_feature, item_feature)
one_train(Data, opt)







