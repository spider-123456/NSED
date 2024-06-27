#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
import random
import json
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, self.behavior_dict['all'][str(idx + 1)]):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)
        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.device = args.device
        self.model_name = args.model_name

        self.__get_count()
        self.__get_behavior_items()
        # self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()

        # self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    #count the number of users and items
    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']


    def __get_behavior_items(self):
        """
        dict中每个用户交互的有哪些item
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    # def __get_validation_dict(self):
    #     """
    #     load the list of items that the user has interacted with in the validation set
    #     :return:
    #     """
    #     with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
    #         b_dict = json.load(f)
    #         self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        load graphs
        :return:
        """
        self.edge_index = {}
        self.behaviors_item = {}
        self.behaviors_user = {}
        self.adj_matrix = {}
        self.adj_norm_matrix = {}
        self.laplacian_matrix = {}
        self.user_behaviour_degree = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))

                #array stack to get the indices
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)
                values = torch.ones(len(row), dtype=torch.float32)
                self.interaction_matrix = self._create_sparse_matrix(row, col,values)
                self.adj_norm_matrix[behavior] = self.get_norm_adj_mat().to(self.device)
                self.laplacian_matrix[behavior] = self.get_laplacian_matrix().to(self.device)
                self.user_behaviour_degree.append(torch.sparse.FloatTensor(indices,values,[self.user_count + 1, self.item_count + 1]).to_dense().sum(dim=1).view(-1, 1))
                #构建边索引
                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1) # 将行索引转换为一维张量
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.adj_matrix[behavior] = torch.sparse.FloatTensor(edge_index, torch.ones(len(edge_index[0])), torch.Size([self.user_count+self.item_count+2, self.user_count+self.item_count+2]))
                self.edge_index[behavior] = edge_index
                self.behaviors_user[behavior] = edge_index[0]
                self.behaviors_item[behavior] = edge_index[1]
        # self.draw_degree(self.adj_matrix)
        self.user_behaviour_degree = torch.cat(self.user_behaviour_degree, dim=1)
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])


    def draw_degree(self,user_item_matrix):
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # 将稀疏张量转换为 COO 格式以获取行和列索引
        color = ['yellow','green','red','blue']
        plt.figure(figsize=(7, 4))
        plt.xlim(0, 350)
        for i, behavior in enumerate(self.behaviors):
            row_indices, col_indices = user_item_matrix[behavior].coalesce().indices().numpy()
            # 计算用户和项目的度
            user_degrees = np.bincount(row_indices)
            item_degrees = np.bincount(col_indices)
            # 将度分组为区间并统计每个区间的用户和项目数量
            # degree_bins = np.arange(0, 150, 1)
            degree_bins = np.arange(0, max(np.max(user_degrees), np.max(item_degrees)) + 1, 2)
            user_degree_hist, _ = np.histogram(user_degrees, bins=degree_bins)
            item_degree_hist, _ = np.histogram(item_degrees, bins=degree_bins)
            # 绘制用户度分布
            plt.plot(degree_bins[:-1]+1, user_degree_hist, color=color[i],label=f'{behavior}')
            plt.fill_between(degree_bins[:-1]+1, user_degree_hist, color=color[i], alpha=0.3)
        plt.legend()
        plt.xlabel('Number of node interactions')
        plt.ylabel('Number of Nodes')
        plt.title(self.model_name)
        plt.tight_layout()
        plt.show()

    def get_norm_adj_mat(self):

        # build adj matrix
        A = sp.dok_matrix(
            (self.user_count + self.item_count+2, self.user_count + self.item_count+2), dtype=np.float32
        )

        inter_m = self.interaction_matrix
        inter_m_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_m.row, inter_m.col + self.user_count+1), [1] * inter_m.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_m_t.row + self.user_count+1, inter_m_t.col),
                    [1] * inter_m_t.nnz,
                )
            )
        )
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_laplacian_matrix(self):
        r"""Get the laplacian matrix of users and items.

        .. math::
            L = I - D^{-1} \times A

        Returns:
            Sparse tensor of the laplacian matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.user_count + self.item_count+2, self.user_count + self.item_count+2), dtype=np.float32
        )

        inter_m = self.interaction_matrix
        inter_m_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_m.row, inter_m.col + self.user_count+1), [1] * inter_m.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_m_t.row + self.user_count+1, inter_m_t.col),
                    [1] * inter_m_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        D = sp.diags(diag)
        A_tilde = D * A

        # covert norm_adj matrix to tensor
        A_tilde = sp.coo_matrix(A_tilde)
        row = A_tilde.row
        col = A_tilde.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_tilde.data)
        A_tilde = torch.sparse.FloatTensor(i, data, torch.Size(A_tilde.shape))
        # generate laplace matrix
        L = self.get_eye_mat(self.user_count + self.item_count+2) - A_tilde
        return L
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


    def _create_sparse_matrix(
        self, source_field, target_field,data
    ):
        src = source_field
        tgt = target_field

        mat = coo_matrix(
            (data, (src, tgt)), shape=(self.user_count+1, self.item_count+1)
        )
        return mat

    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    # def validate_dataset(self):
    #     return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5)
    # [batch_size,behaviors,3] 不同行为的[user,pos,neg]
    for index , item in enumerate(loader):
        if(index==0):
            print(item,item.shape)


