#!/usr/bin/env
# coding:utf-8
"""
Created on 2019/11/9 11:07

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import csv
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
current_path = osp.dirname(osp.realpath(__file__))
# every_5weeks_file = 'processed/every_5_weeks.csv'
# cnty_file = 'processed/df_cnty_selected.csv'
# every_52weeks_file = 'processed/every_53_weeks_raw.csv'
# sorted_every_52weeks_file = 'processed/sorted_every_53_weeks_raw.csv'

max_ft_size = 12
max_pred_size = 6
china_ratio =  'global_flu/China_ILI.xlsx'
# us_ratio = 'global_flu/us_ratio.csv'
us_ratio = 'us_flu/us_ratio.csv'


class GlobalFlu(object):
    def __init__(self, data_type='us', split_param=[0.6, 0.2, 0.2], wind_size=12, pred_step=1, ratio=100):
        self.ft_mat = []
        self.label_mat = []
        self.row_name_list = []
        self.node_size = 0
        self.node_feature_size = 0
        self.label_size = 1
        self.data_type = data_type

        if self.data_type == 'us':
            self.load_us_ratio_multi_step(wind_size=wind_size, pred_step=pred_step)
        else:
            self.load_china_ratio_multi_step(wind_size=wind_size, pred_step=pred_step, data_type=data_type)
        self.data_ratio_change(ratio)
        self.train_test_split(split_param,  mode='ratio', shuffle=False)

    def to_device(self, device):
        self.ft_mat = self.ft_mat.to(device)
        self.label_mat = self.label_mat.to(device)

    def to_tensor(self):
        self.ft_mat = torch.FloatTensor(self.ft_mat)
        self.label_mat = torch.FloatTensor(self.label_mat)

    def data_ratio_change(self, ratio):
        self.ft_mat = self.ft_mat * ratio
        self.label_mat = self.label_mat * ratio
    #
    # def load_us_ratio(self, ft_size=52):
    #     file_path = osp.join(current_path, us_ratio)
    #     df = pd.read_csv(file_path)
    #     data_mat = df.to_numpy()
    #     label = data_mat[:, -1]
    #     node_mat = data_mat[:, 53 - ft_size:53]
    #     # print(label)
    #     # print(node_mat[0,:])
    #     self.label_mat = np.array(label)
    #     self.ft_mat = np.array(node_mat)
    #     self.node_size = self.ft_mat.shape[0]
    #     self.node_feature_size = self.ft_mat.shape[1]

    def load_us_ratio_multi_step(self, wind_size=52, pred_step=1):
        file_path = osp.join(current_path, us_ratio)
        # file_path = osp.join(current_path, every_52weeks_file)
        df = pd.read_csv(file_path)
        data_mat = df.to_numpy()
        # self.ft_mat = data_mat[:, 53 - wind_size:53]
        self.ft_mat = data_mat[:, -wind_size-1:-1]
        one_step_label = data_mat[:, -1]
        self.label_mat = []
        for idx in range(0, one_step_label.shape[0]):
            tmp_label = list(one_step_label[idx: idx + pred_step])
            self.label_mat.append(tmp_label)
        # print(label)
        if pred_step > 1:
            self.ft_mat = self.ft_mat[:-(pred_step-1)]
            self.label_mat = self.label_mat[:-(pred_step-1)]
            # self.ft_mat = node_mat[0: all_weeks_num - label_step + 1]
            # self.label_mat = label[0: all_weeks_num - label_step + 1]

        self.ft_mat = np.array(self.ft_mat).reshape(-1, wind_size)
        self.label_mat = np.array(self.label_mat).reshape(-1, pred_step)
        self.node_size = self.ft_mat.shape[0]
        self.node_feature_size = self.ft_mat.shape[1]
        self.label_size = self.label_mat.shape[1]

    def load_china_ratio_multi_step(self, wind_size=6, pred_step=1, data_type='ch_north'):
        '''
        :param ft_size:
        :param label_step:
        :param type: ch_north or ch_south
        :return:
        '''
        data_df = pd.read_excel(osp.join(current_path, china_ratio))
        # print(data_df)
        if data_type == 'ch_south':
            ili_rate = data_df['ILI rate（South）'].to_list()
        elif data_type == 'ch_north':
            ili_rate = data_df['ILI rate（North）'].to_list()
        else:
            exit('error china data type')

        # 缺失值 使用前一时刻填充
        for idx in range(len(ili_rate)):
            current_rate = ili_rate[idx]
            if current_rate == '-' or math.isnan(current_rate):
                ili_rate[idx] = float(ili_rate[idx - 1])
            # print(type(current_rate))
            # print(ili_rate[idx])
        # print(ili_rate)
        ili_rate = np.array(ili_rate).astype(np.float)


        node_ft_mat = []
        label = []
        for idx in range(max_ft_size, ili_rate.shape[0] - max_pred_size):
            item_ft = ili_rate[idx-wind_size :idx]
            item_label = ili_rate[idx: idx + pred_step]
            node_ft_mat.append(item_ft)
            label.append(item_label)

        self.ft_mat = np.array(node_ft_mat)
        self.label_mat = np.array(label)

        self.label_size = self.label_mat.shape[1]
        self.node_feature_size = self.ft_mat.shape[1]
        self.node_size = self.ft_mat.shape[0]




    def train_test_split(self, split_param, mode='ratio', shuffle=False):
        '''
        :param param: The proportion or number of data divided
        :param mode: 'ratio' or 'numerical'
        :return:
        '''
        if mode == 'ratio' and sum(split_param) <= 1:
            split_param = np.array(split_param)
            train_size, valid_size, test_size = map(int, np.floor(split_param * self.node_size))

        elif mode == 'numerical' or sum(split_param) > 1:
            split_param = np.array(split_param)
            # print(np.sum(split_param), self.node_size)
            assert np.sum(split_param) <= self.node_size
            train_size, valid_size, test_size = split_param


        node_index = np.arange(0, self.node_size)
        if shuffle is True:
            np.random.shuffle(node_index)

        self.train_index = node_index[0:train_size]
        self.valid_index = node_index[train_size: valid_size + train_size]
        # self.test_index = node_index[valid_size + train_size: valid_size + train_size + test_size]
        self.test_index = node_index[valid_size + train_size: valid_size + train_size + test_size]

        return True


if __name__ == '__main__':
    demo = GlobalFlu(pred_step=3, wind_size=12, split_param=[0.6, 0.2, 0.2], data_type='us')
    print(demo.ft_mat.shape)

