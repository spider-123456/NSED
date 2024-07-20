#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model_cascade.py
# @Author: Wcai
# @Date  : 2024/3/26 16:16
# @Desc  : NESD

import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from data_set_beibei import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss

#FOR NGCF
class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)
#FOR NGCF
class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2
class GraphEncoder(nn.Module):  #GCN
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        embeddings_list = [x]
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            embeddings_list.append(x)
        return x, embeddings_list

class LightGCN(nn.Module):
    def __init__(self, layers):
        super(LightGCN, self).__init__()
        self.layers = layers

    def forward(self, ItemAndUserEmebddings,adj_matrix):
        """
            input: allEmebeddings
            ouput: the updated embeddings of user item
        """
        all_embeddings = ItemAndUserEmebddings
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings,embeddings_list
class SpectralCF(nn.Module):
    def __init__(self, layers,emb_dim):
        super(SpectralCF, self).__init__()
        self.layers = layers
        self.emb_dim = emb_dim
        self.sigmoid = torch.nn.Sigmoid()
        self.filters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.normal(
                        mean=0.01, std=0.02, size=(self.emb_dim, self.emb_dim)
                    ),
                    requires_grad=True,
                )
                for _ in range(self.layers)
            ]
        )
    def forward(self,ItemAndUserEmebddings, A_hat):
        all_embeddings = ItemAndUserEmebddings
        embeddings_list = [all_embeddings]
        for k in range(self.layers):
            all_embeddings = torch.sparse.mm(A_hat, all_embeddings)
            all_embeddings = self.sigmoid(torch.mm(all_embeddings, self.filters[k]))
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings,embeddings_list


class NGCF(nn.Module):
    def __init__(self,emb_dim,hidden_size_list,node_dropout,message_dropout):
        super(NGCF, self).__init__()
        # load parameters info
        self.embedding_size = emb_dim
        self.hidden_size_list = hidden_size_list
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.sparse_dropout = SparseDropout(self.node_dropout)

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
                zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))


    def forward(self,ItemAndUserEmebddings, norm_adj_matrix,eye_matrix):
        A_hat = (
            self.sparse_dropout(norm_adj_matrix)
            if self.node_dropout != 0
            else norm_adj_matrix
        )
        all_embeddings = ItemAndUserEmebddings
        embeddings_list = [all_embeddings]

        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings,embeddings_list


class NSED(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(NSED, self).__init__()

        self.device = args.device
        self.layers = args.layers

        # self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.n_users = dataset.user_count
        self.n_items = dataset.item_count

        self.edge_index = dataset.edge_index
        self.adj_matrix = dataset.adj_matrix
        self.adj_norm_matrix = dataset.adj_norm_matrix
        self.laplacian_matrix = dataset.laplacian_matrix
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.batch_size = args.batch_size
        self.head_num = args.head_num
        self.att_dim = args.att_dim
        self.d_h = self.embedding_size // self.head_num
        self.all_weights = {}
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self._item = dataset.behaviors_item
        self._user = dataset.behaviors_user
        self.drop_ratio = args.drop_ratio
        self.temp = args.temp
        self.ssl_tau = args.ssl_tau
        self.type = args.type
        self.ssl_weight = args.ssl_weight
        # NGCF Propertites ---------------
        self.hidden_size_list = args.hidden_size_list
        self.node_dropout = args.node_dropout
        self.message_dropout = args.message_dropout

        self.all_weights['beta'] = nn.Parameter(torch.FloatTensor(len(self.behaviors), self.embedding_size,self.embedding_size)).to(self.device)
        self.all_weights['Wq'] = Parameter(
            torch.FloatTensor(len(self.behaviors), self.embedding_size,self.head_num * self.d_h)).to(self.device)
        self.all_weights['Wk'] = Parameter(
            torch.FloatTensor(len(self.behaviors), self.embedding_size, self.head_num * self.d_h)).to(self.device)
        self.all_weights['Wv'] = Parameter(
            torch.FloatTensor(len(self.behaviors), self.embedding_size, self.head_num * self.d_h)).to(self.device)

        self.Graph_encoder = nn.ModuleDict({
            # gcn --can not run kmean
            # behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
            #lightgcn
            behavior: LightGCN(self.layers[index]) for index, behavior in enumerate(self.behaviors)
            #SpectralCF
            # behavior: SpectralCF(self.layers[index],self.embedding_size) for index, behavior in enumerate(self.behaviors)
            #NGCF
            # behavior: NGCF(self.embedding_size,self.hidden_size_list,self.node_dropout,self.message_dropout) for index, behavior in enumerate(self.behaviors)
        })


        self.k = args.num_clusters
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.ssl_reg = args.ssl_reg
        self.proto_reg = args.proto_reg
        self.inter_reg = args.inter_reg
        self.m_step = args.m_step
        self.hyper_layers = args.hyper_layers
        self.gama = args.gama
        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.run_ncl_ssl = args.run_ncl_ssl
        self.run_ncl_proto = args.run_ncl_proto
        self.run_sgl = args.run_sgl
        self.run_inter = args.run_inter
        self.task_weight = args.task_weight

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.storage_all_embeddings = None
        self.apply(self._init_weights)
        self._load_model()



        # self.sub_graph1[behavior] = sub_graph1_temp
        # self.sub_graph2[behavior] = sub_graph2_temp
    #每一个行为都进行一次图重构.
    def graph_construction(self,behaviors):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node."""
        self.sub_graph1 = {}
        self.sub_graph2 = {}
        for index,behavior in enumerate(behaviors):
            sub_graph1_temp = []
            sub_graph2_temp = []
            if self.type == "ND" or self.type == "ED":
                sub_graph1_temp = self.csr2tensor(self.create_adjust_matrix(is_sub=True,behavior=behavior))
            elif self.type == "RW":
                for i in range(self.layers[index]):
                    _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                    sub_graph1_temp.append(_g)

            if self.type == "ND" or self.type == "ED":
                sub_graph2_temp = self.csr2tensor(self.create_adjust_matrix(is_sub=True,behavior=behavior))
            elif self.type == "RW":
                for i in range(self.n_layers):
                    _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True,behavior=behavior))
                    sub_graph2_temp.append(_g)

            self.sub_graph1[behavior] = sub_graph1_temp
            self.sub_graph2[behavior] = sub_graph2_temp

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        nn.init.xavier_uniform_(self.all_weights['Wk'])
        nn.init.xavier_uniform_(self.all_weights['Wq'])
        nn.init.xavier_uniform_(self.all_weights['Wv'])
        nn.init.xavier_uniform_(self.all_weights['beta'])



    def gcn_propagate(self,graph_adj_matrix):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        all_layer_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        for index,behavior in enumerate(self.behaviors):
            layer_embeddings = total_embeddings
            indices = self.edge_index[behavior].to(self.device)
            # GCN
            # layer_embeddings, all_layer_embedding = self.Graph_encoder[behavior](layer_embeddings, indices)
            #LightGCN
            layer_embeddings, all_layer_embedding = self.Graph_encoder[behavior](layer_embeddings, graph_adj_matrix[behavior])

            # spectralcf
            # I = self.get_eye_mat(self.n_items + self.n_users+2).to(self.device)
            # L = self.laplacian_matrix[behavior].to(self.device)
            # A_hat = I + L
            # layer_embeddings, all_layer_embedding = self.Graph_encoder[behavior](layer_embeddings, A_hat)

            # NGCF
            # forward(self, ItemAndUserEmebddings, norm_adj_matrix, eye_matrix):
            # eye_matrix = self.get_eye_mat(self.n_items + self.n_users + 2).to(self.device)
            # layer_embeddings, all_layer_embedding = self.Graph_encoder[behavior](layer_embeddings, graph_adj_matrix[behavior], eye_matrix)

            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            # total_embeddings = layer_embeddings+ total_embeddings
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings
            all_layer_embeddings[behavior] = all_layer_embedding
        return all_embeddings, all_layer_embeddings


    def forward(self, batch_data):
        self.storage_all_embeddings = None
        all_embeddings, embeddings_list = self.gcn_propagate(self.adj_norm_matrix)
        if(self.run_sgl):
            sub_graph1_all_embeddings, sub_graph1_embeddings_list = self.gcn_propagate(self.sub_graph1)
            sub_graph2_all_embeddings, sub_graph2_embeddings_list = self.gcn_propagate(self.sub_graph2)
        total_loss = 0
        ssl_loss_ncl, ssl_loss_ncl1,proto_loss, subgraph_loss, inter_loss = 0, 0, 0, 0, 0
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
            if(self.run_sgl):
                # 子图损失函数计算方式1：sgl_loss
                user_sub1, item_sub1 = torch.split(sub_graph1_all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
                user_sub2, item_sub2 = torch.split(sub_graph2_all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
                subgraph_loss = self.calc_ssl_sgl_loss(users, items[:,0], user_sub1, user_sub2, item_sub1, item_sub2)

            center_embedding = embeddings_list[behavior][0]
            # context_embedding = embeddings_list[behavior][self.hyper_layers * 2]
            context_embedding = embeddings_list[behavior][self.hyper_layers]
            # layer = 1  take the  init embedding as the neighbor of 1-th embedding
            if(self.layers[index] == 1):
                center_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
                context_embedding = embeddings_list[behavior][self.hyper_layers]
            # layer = 2  take the 2-th output embedding as the neighbor of 1-th embedding
            elif(self.layers[index] == 2):
                center_embedding = embeddings_list[behavior][0]
                context_embedding = embeddings_list[behavior][self.hyper_layers]
            # layer = 3  take the 3-th output embedding as the neighbor of 1-th embedding
            elif(self.layers[index] == 3):
                center_embedding = embeddings_list[behavior][0]
                context_embedding = embeddings_list[behavior][self.hyper_layers * 2]
            elif(self.layers[index] == 4):
                center_embedding = embeddings_list[behavior][0]
                context_embedding = embeddings_list[behavior][self.hyper_layers * 2]

                center_embedding_1 = embeddings_list[behavior][0+1]
                context_embedding_1 = embeddings_list[behavior][self.hyper_layers * 2+1]
                if (self.run_ncl_ssl):
                    ssl_loss_ncl1 = self.ssl_layer_loss(context_embedding_1, center_embedding_1, users, items[:, 0])


            # if(self.run_ncl_proto):
            #     proto_loss = self.ProtoNCE_loss(center_embedding, users, items[:, 0])

            if(self.run_ncl_ssl):
                ssl_loss_ncl = self.ssl_layer_loss(context_embedding, center_embedding, users, items[:, 0])
            ncl_loss = ssl_loss_ncl + proto_loss + ssl_loss_ncl1

            #我需要对不同行为图聚集的user-item兴趣进行一个对比，并计算出对比损失。
            if(self.run_inter):
                auxiliary_user, auxiliary_item = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
                target_user, target_item = torch.split(all_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
                inter_loss = self.calc_inter_loss(users, items[:, 0],auxiliary_user, target_user, auxiliary_item, target_item)

            # BPR_LOSS
            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]
            # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            # total_loss = total_loss + self.bpr_loss(scores[:, 0], scores[:, 1]) + ncl_loss + subgraph_loss + inter_loss
            # total_loss = total_loss + self.bpr_loss(scores[:, 0], scores[:, 1]) + ncl_loss + subgraph_loss + inter_loss
            total_loss = total_loss + self.task_weight[index] * (self.bpr_loss(scores[:, 0], scores[:, 1]) + ncl_loss + subgraph_loss + inter_loss)
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        return total_loss

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def calculate_ncl_loss(self,center_embedding,context_embedding,users,items,all_embeddings):
        ssl_loss_ncl = self.ssl_layer_loss(
            context_embedding, center_embedding, users, items[:, 0]
        )
        # 原型损失
        # proto_loss_ncl = self.ProtoNCE_loss(center_embedding, users, items[:, 0])
        proto_loss_ncl = self.ProtoNCE_loss(all_embeddings, users, items[:, 0])

        return ssl_loss_ncl+proto_loss_ncl



    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            all_embeddings, _ = self.gcn_propagate(self.adj_norm_matrix)
            # self.storage_all_embeddings = self.agg_embedding(all_embeddings)
            self.storage_all_embeddings = all_embeddings[self.behaviors[-1]]
        # user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
        user_embedding, item_embedding = torch.split(self.storage_all_embeddings, [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores

    def agg_embedding(self,all_embeddings):
        temp = []
        for index,behavior in enumerate(self.behaviors):
            temp.append(all_embeddings[behavior])

        return torch.sum(torch.stack(temp,dim=1),dim=1)



    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users+1, self.n_items+1]
        )
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(
            previous_embedding, [self.n_users+1, self.n_items+1]
        )

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.temp)
        ttl_score_user = torch.exp(ttl_score_user / self.temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.temp)
        ttl_score_item = torch.exp(ttl_score_item / self.temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.gama * ssl_loss_item)
        return ssl_loss


    def calc_ssl_sgl_loss(
        self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
    ):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 /self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))
        return (ssl_item + ssl_user) * self.ssl_weight
    def calc_inter_loss(
        self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
    ):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.temp)
        v2 = torch.sum(torch.exp(v2 / self.temp), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.temp)
        v4 = torch.sum(torch.exp(v4 / self.temp), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))
        return (ssl_item + ssl_user) * self.inter_reg

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)



    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)),
            matrix.shape,
        ).to(self.device)
        return x
    def get_eye_mat(self, num):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Args:
            num: number of column of the square matrix

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)
    def create_adjust_matrix(self, is_sub,behavior):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user[behavior], dtype=np.float32)
            matrix = sp.csr_matrix(
                (ratings, (self._user[behavior], self._item[behavior] + self.n_users)),
                shape=(self.n_users + self.n_items, self.n_users + self.n_items),
            )
        else:
            if self.type == "ND":
                drop_user = self.rand_sample(
                    self.n_users,
                    size=int(self.n_users * self.drop_ratio),
                    replace=False,
                )
                drop_item = self.rand_sample(
                    self.n_items,
                    size=int(self.n_items * self.drop_ratio),
                    replace=False,
                )
                R_user = np.ones(self.n_users, dtype=np.float32)
                R_user[drop_user] = 0.0
                R_item = np.ones(self.n_items, dtype=np.float32)
                R_item[drop_item] = 0.0
                R_user = sp.diags(R_user)
                R_item = sp.diags(R_item)
                R_G = sp.csr_matrix(
                    (
                        np.ones_like(self._user[behavior], dtype=np.float32),
                        (self._user[behavior], self._item[behavior]),
                    ),
                    shape=(self.n_users, self.n_items),
                )
                res = R_user.dot(R_G)
                res = res.dot(R_item)

                user, item = res.nonzero()
                ratings = res.data
                matrix = sp.csr_matrix(
                    (ratings, (user, item )),
                    shape=(self.n_users + self.n_items+2, self.n_users + self.n_items+2),
                )

            elif self.type == "ED" or self.type == "RW":
                keep_item = self.rand_sample(
                    len(self._user[behavior]),
                    size=int(len(self._user[behavior]) * (1 - self.drop_ratio)),
                    replace=False,
                )
                user = self._user[behavior][keep_item]
                item = self._item[behavior][keep_item]

                matrix = sp.csr_matrix(
                    (np.ones_like(user), (user, item)),
                    shape=(self.n_users + self.n_items+2, self.n_users + self.n_items+2),
                )
        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample
