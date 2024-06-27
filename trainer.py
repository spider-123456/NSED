#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author: yanms
# @Date  : 2021/11/1 15:45
# @Desc  :
import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_set import DataSet
from metrics import metrics_dict

class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.train_writer = args.train_writer
        self.test_writer = args.test_writer
        self.device = args.device
        self.warm_up_step = args.warm_up_step
        self.TIME = args.TIME
        self.optimizer = self.get_optimizer(self.model)
        self.num_m_step = args.m_step
        self.run_ncl_ssl = args.run_ncl_ssl
        self.run_ncl_proto = args.run_ncl_proto
        self.run_sgl = args.run_sgl

    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer



    @logger.catch()
    def train_model(self):
        # [batch_size,behaviors,3]
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        best_result = 0
        best_dict = {}
        best_epoch = 0
        best_model = None
        final_test = None

        # 首先进行图的构造，构造出两个子图，如何在原图以及这两个子图上进行
        if(self.run_sgl):
            self.model.graph_construction(self.behaviors)
        for epoch in range(self.epochs):
            # # only differences from the original trainer
            if epoch % self.num_m_step == 0 and self.run_ncl_proto:
                self.model.e_step()
            test_metric_dict = self._train_one_epoch(train_dataset_loader, epoch)
            result = test_metric_dict['ndcg@10']
            # early stop
            if result - best_result > 0:
                final_test = test_metric_dict
                best_result = result
                best_dict = test_metric_dict
                # best_model_state_dict = copy.deepcopy(self.model)
                best_model = self.model.state_dict()
                best_epoch = epoch
            if epoch - best_epoch > 20:
                break
            # save the best model
            self.save_model(best_model)

        # # save the best model
        # self.save_model(best_model)
        logger.info(f"training end, best iteration %d, results: %s" %
                    (best_epoch + 1, best_dict.__str__()))
        logger.info(f"final test result is:  %s" % final_test.__str__())

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        self.model.train()
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )

        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no
        self.train_writer.add_scalar('total Train loss', total_loss, epoch + 1)
        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        # test
        start_time = time.time()
        # def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer):
        test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_interacts, self.dataset.test_gt_length,
                                         self.test_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"test %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

        return test_metric_dict

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            start = time.time()
            scores = self.model.full_predict(batch_data)
            for index, user in enumerate(batch_data):
                #计算出了这个user 与其他所有Item的交互得分。
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    # 是一个特殊值，表示负无穷大，通常用于表示一个值比任何有限值都小的情况,可以作为数值与其他进行比较
                    user_score[items] = -np.inf
                #item 索引将不会参与到下面的topk排序中去。所有的item中查找，
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch + 1)
        return metric_dict

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")
    def calculate_result(self, topk_list, gt_len):
        result_list = []

        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)

        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, state_dict):
        torch.save(state_dict, os.path.join(self.model_path, self.model_name + self.TIME + '.pth'))
