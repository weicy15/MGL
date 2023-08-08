import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
import torch.nn.init as init
import numpy as np
from util import load_data
from collections import defaultdict
import pandas as pd
from copy import deepcopy
import torch_sparse
import random


def sigmoid(x, k):
    s = 1 - (k / (k + np.exp(x/k)))
    return s

def inverse_sigmoid(x, k):
    s = k / (k + np.exp(x/k))
    return s


class Generator(nn.Module):
    def __init__(self, user_num, item_num, item_feature_list, item_feature_matrix, dense_f_list_transforms, opt, device):
        super(Generator, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.item_feature_list = deepcopy(item_feature_list)
        self.item_feature_matrix = item_feature_matrix.to(device)

        self.item_dense_features = []
        for dense_f in dense_f_list_transforms.values():
            self.item_dense_features.append(dense_f.to(device))

        self.item_feature_list.remove({'feature_name':'encoded', 'feature_dim':self.item_num})

        item_embedding_dims = defaultdict(int)
        for f in self.item_feature_list:
            item_embedding_dims[f['feature_name']] = opt.embedding_size

        self.item_total_emb_dim = sum(list(item_embedding_dims.values())) + opt.dense_embedding_dim * len(self.item_dense_features)

        self.item_Embeddings = nn.ModuleList([nn.Embedding(feature['feature_dim'], item_embedding_dims[feature['feature_name']]) for feature in self.item_feature_list])

        self.item_dense_Embeddings = nn.ModuleList([nn.Linear(dense_f.shape[1], opt.dense_embedding_dim, bias=False) for dense_f in self.item_dense_features])

        self.encoder = nn.Sequential(nn.Linear(self.item_total_emb_dim, 256, bias=True), nn.ReLU(), nn.Linear(256, 64, bias=True), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 256, bias=True), nn.ReLU(), nn.Linear(256, opt.id_embedding_size, bias=True))


    def encode(self, item_id):
        batch_item_feature_embedded = self.embed_feature(item_id)

        batch_item_feature_encoded = self.encoder(batch_item_feature_embedded)

        return batch_item_feature_encoded

    def decode(self, batch_item_feature_encoded):
        pre_item_id_embedded = self.decoder(batch_item_feature_encoded)
        return pre_item_id_embedded


    def embed_feature(self, item_id):
        batch_item_feature_embedded = []
        batch_item_feature  = self.item_feature_matrix[item_id]
        for i, f in enumerate(self.item_feature_list):
            embedding_layer = self.item_Embeddings[i]
            batch_item_feature_i = batch_item_feature[:, i]
            batch_item_feature_i_embedded = embedding_layer(batch_item_feature_i)

            batch_item_feature_embedded.append(batch_item_feature_i_embedded)

        batch_item_feature_embedded = torch.cat(batch_item_feature_embedded, -1)

        dense_embeddings = []
        for i, dense_f in enumerate(self.item_dense_features):
            batch_dense_f = dense_f[item_id]
            dense_embedded = self.item_dense_Embeddings[i](batch_dense_f.float()) / torch.sum(batch_dense_f.float(), dim = 1, keepdim= True)
            dense_embeddings.append(dense_embedded)

        batch_item_feature_embedded = torch.cat([batch_item_feature_embedded] + dense_embeddings, dim=1)

        return batch_item_feature_embedded

class Model(nn.Module):
    def __init__(self, Data, opt, device):
        super(Model, self).__init__()

        self.name = "Meta_final_2"

        self.interact_train = Data.interact_train

        self.user_num = Data.user_num
        self.item_num = Data.item_num
        self.item_feature_list = Data.item_feature_list
        self.item_feature_matrix = Data.item_feature_matrix
        self.dense_f_list_transforms = Data.dense_f_list_transforms

        self.generator = Generator(self.user_num, self.item_num, self.item_feature_list, self.item_feature_matrix, self.dense_f_list_transforms, opt, device)

        self.user_id_Embeddings = nn.Embedding(self.user_num, opt.id_embedding_size)
        self.item_id_Embeddings = nn.Embedding(self.item_num, opt.id_embedding_size)

        self.device = device

        self.L = opt.L
        self.link_topk = opt.link_topk

        self.userDegrees = Data.userDegrees 
        self.itemDegrees = Data.itemDegrees

        sorted_item_degrees = sorted(self.itemDegrees.items(), key=lambda x: x[0])
        _, item_degree_list = zip(*sorted_item_degrees)
        self.item_degree_numpy = np.array(item_degree_list)

        self.create_sparse_adjaceny()

        self.top_rate = opt.top_rate
        self.convergence = opt.convergence



    def create_sparse_adjaceny(self):
        index = [self.interact_train['userid'].tolist(), self.interact_train['itemid'].tolist()]
        value = [1.0] * len(self.interact_train)

        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.user_num, self.item_num)).to(self.device)

        tmp_index = [self.interact_train['userid'].tolist(), (self.interact_train['itemid'] + self.user_num).tolist()]
        tmp_adj = torch.sparse_coo_tensor(tmp_index, value, (self.user_num+self.item_num, self.user_num+self.item_num))
        
        joint_adjaceny_matrix = (tmp_adj + tmp_adj.t()).coalesce()

        row_indices = joint_adjaceny_matrix.indices()[0]
        col_indices = joint_adjaceny_matrix.indices()[1]
        joint_adjaceny_matrix_value = joint_adjaceny_matrix.values()


        degree = torch.sparse.sum(joint_adjaceny_matrix, dim=1).to_dense()
        degree = torch.pow(degree, -1)
        degree[torch.isinf(degree)] = 0

        self.joint_adjaceny_matrix = joint_adjaceny_matrix.to(self.device)

        joint_adjaceny_matrix_normal_value = degree[row_indices] * joint_adjaceny_matrix_value
        self.joint_adjaceny_matrix_normal_spatial = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices], dim=0), joint_adjaceny_matrix_normal_value, (self.user_num+self.item_num, self.user_num+self.item_num)).to(self.device)
        

    def link_predict(self, itemDegrees, top_rate):

        sorted_item_degrees = sorted(itemDegrees.items(), key=lambda x: x[1])
        item_list_sorted, d_item = zip(*sorted_item_degrees)
        item_tail = torch.tensor(item_list_sorted).to(self.device)

        top_length = int(self.item_num * top_rate)
        item_top = torch.tensor(item_list_sorted[-top_length:]).to(self.device)


        top_item_embedded = self.generator.encode(item_top)
        tail_item_embedded = self.generator.encode(item_tail)
        
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))

        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim= -1)
        i2i_score_masked = i2i_score_masked.sigmoid()


        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        top_item_degree = torch.sum(i2i_score_masked, dim=0)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        top_item_degree = torch.pow(top_item_degree + 1, -1).unsqueeze(0).expand_as(i2i_score_masked).reshape(-1)


        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)

        row_index = (tail_item_index+self.user_num).unsqueeze(0)
        colomn_index = (top_item_index+self.user_num).unsqueeze(0)
        joint_enhanced_value = enhanced_value * tail_item_degree
        
        return row_index, colomn_index, joint_enhanced_value



    def i2i(self, item1, item2):

        mse_loss = nn.MSELoss()
        item1_embedded = self.generator.encode(item1)
        item2_embedded = self.generator.encode(item2)

        item_list = list(range(self.item_num))
        random.shuffle(item_list)
        item2_num = item2_embedded.shape[0]
        item_false = torch.tensor(item_list[:item2_num]).to(self.device)

        item_false_embedded = self.generator.encode(item_false)

        i2i_score = torch.mm(item1_embedded, item2_embedded.permute(1, 0)).sigmoid()
        i2i_score_false = torch.mm(item1_embedded, item_false_embedded.permute(1, 0)).sigmoid()

        loss = (mse_loss(i2i_score, torch.ones_like(i2i_score)) + mse_loss(i2i_score_false, torch.zeros_like(i2i_score_false))) / 2

        return loss



    def reg(self, pos_item):
        pos_item_encoded = self.generator.encode(pos_item)
        pos_item_decoded = self.generator.decode(pos_item_encoded)

        pos_item_degree = self.item_degree_numpy[pos_item.cpu().numpy()]
        probs = sigmoid(pos_item_degree, self.convergence)


        def ssl_compute(normalized_embedded_s1, normalized_embedded_s2, probs):
            # batch_size
            pos_score = torch.sum(torch.mul(normalized_embedded_s1, normalized_embedded_s2), dim=1, keepdim=False)
            # batch_size * batch_size
            all_score = torch.mm(normalized_embedded_s1, normalized_embedded_s2.t())
            ssl_mi = (probs * torch.log(torch.exp(pos_score/self.ssl_temp) / torch.exp(all_score/self.ssl_temp).sum(dim=1, keepdim=False))).mean()
            return ssl_mi

        user_embeddings, item_embeddings = self.compute_embeddings()
        reg_loss = ssl_compute(pos_item_decoded, item_embeddings[pos_item], probs)

        return reg_loss


    def q_link_predict(self, itemDegrees, top_rate, fast_weights):

        sorted_item_degrees = sorted(itemDegrees.items(), key=lambda x: x[1])
        item_list_sorted, d_item = zip(*sorted_item_degrees)
        item_tail = torch.tensor(item_list_sorted).to(self.device)

        top_length = int(self.item_num * top_rate)
        item_top = torch.tensor(item_list_sorted[-top_length:]).to(self.device)


        encoder_0_weight = fast_weights[0]
        encoder_0_bias = fast_weights[1]
        encoder_2_weight = fast_weights[2]
        encoder_2_bias = fast_weights[3]

        top_item_feature = self.generator.embed_feature(item_top)
        tail_item_feature = self.generator.embed_feature(item_tail)

        top_item_hidden = torch.mm(top_item_feature, encoder_0_weight.t()) + encoder_0_bias
        top_item_embedded = torch.mm(top_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        tail_item_hidden = torch.mm(tail_item_feature, encoder_0_weight.t()) + encoder_0_bias
        tail_item_embedded = torch.mm(tail_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))

        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim= -1)
        i2i_score_masked = i2i_score_masked.sigmoid()

        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        top_item_degree = torch.sum(i2i_score_masked, dim=0)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        top_item_degree = torch.pow(top_item_degree + 1, -1).unsqueeze(0).expand_as(i2i_score_masked).reshape(-1)


        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)

        row_index = (tail_item_index+self.user_num).unsqueeze(0)
        colomn_index = (top_item_index+self.user_num).unsqueeze(0)
        joint_enhanced_value = enhanced_value * tail_item_degree
        
        return row_index, colomn_index, joint_enhanced_value


    def q_forward(self, user_id, pos_item, neg_item, fast_weights):
        row_index, colomn_index, joint_enhanced_value = self.q_link_predict(self.itemDegrees, self.top_rate, fast_weights)
        indice = torch.cat([row_index, colomn_index], dim=0).to(self.device)

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]
        enhance_weight = torch.from_numpy(inverse_sigmoid(self.item_degree_numpy, self.convergence))
        enhance_weight = torch.cat([torch.zeros(self.user_num), enhance_weight], dim=-1).to(self.device).float()

        for i in range(self.L):
            cur_embedding_ori = torch.mm(self.joint_adjaceny_matrix_normal_spatial, cur_embedding)
            cur_embedding_enhanced = torch_sparse.spmm(indice, joint_enhanced_value, self.user_num + self.item_num, self.user_num + self.item_num, cur_embedding)
            cur_embedding = cur_embedding_ori + enhance_weight.unsqueeze(-1) * cur_embedding_enhanced
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]
        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1, keepdim=False)
        neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1, keepdim=False)

        rec_loss = -(pos_score - neg_score).sigmoid().log().mean()
        
        return rec_loss


    # full item set
    def predict(self, user_id):
        row_index, colomn_index, joint_enhanced_value = self.link_predict(self.itemDegrees, self.top_rate)
        indice = torch.cat([row_index, colomn_index], dim=0).to(self.device)

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]

        enhance_weight = torch.from_numpy(inverse_sigmoid(self.item_degree_numpy, self.convergence))
        enhance_weight = torch.cat([torch.zeros(self.user_num), enhance_weight], dim=-1).to(self.device).float()

        for i in range(self.L):
            cur_embedding_ori = torch.mm(self.joint_adjaceny_matrix_normal_spatial, cur_embedding)
            cur_embedding_enhanced = torch_sparse.spmm(indice, joint_enhanced_value, self.user_num + self.item_num, self.user_num + self.item_num, cur_embedding)
            cur_embedding = cur_embedding_ori + enhance_weight.unsqueeze(-1) * cur_embedding_enhanced
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        user_embedded = user_embeddings[user_id]

        pos_item_embedded = item_embeddings

        score = torch.mm(user_embedded, pos_item_embedded.t())

        return score

    def compute_embeddings(self):
        cur_embedding = torch.cat([self.user_id_Embedding.weight, self.item_id_Embeddings.weight], dim=0)
        all_embeddings = [cur_embedding]

        for i in range(self.L):
            
            cur_embedding = torch.mm(self.joint_adjaceny_matrix_normal, cur_embedding)
            all_embeddings.append(cur_embedding)
        
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        return user_embeddings, item_embeddings