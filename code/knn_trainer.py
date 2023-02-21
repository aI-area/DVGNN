#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/6/3 16:20

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor

from tools.evaluate_utils import evaluate_regression
from dataset.load_dataset import GlobalFlu
from sklearn.multioutput import MultiOutputRegressor


class KNNTrainer(object):
    def __init__(self, data_type='us', split_param=[0.6, 0.2, 0.2], wind_size=12, pred_step=1, seed=3):
        self.setup_seed(seed)

        self.n_neighbors = 4
        self.weights = 'uniform'
        print('KNN  n_neighbors = {}, weights = {}'.format(self.n_neighbors, self.weights))

        self.dataset = GlobalFlu(data_type=data_type, split_param=split_param, wind_size=wind_size,
                                     pred_step=pred_step)
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        # self.start()


    def start(self):
        # multi_step_regression
        multi_target_forest = MultiOutputRegressor(self.model)

        train_ft_mat = self.dataset.ft_mat[self.dataset.train_index]
        train_label_mat = self.dataset.label_mat[self.dataset.train_index]

        valid_ft_mat = self.dataset.ft_mat[self.dataset.valid_index]
        valid_label_mat = self.dataset.label_mat[self.dataset.valid_index]

        test_ft_mat = self.dataset.ft_mat[self.dataset.test_index]
        test_label_mat = self.dataset.label_mat[self.dataset.test_index]

        multi_target_forest.fit(train_ft_mat, train_label_mat)

        train_pred = multi_target_forest.predict(train_ft_mat)

        valid_pred = multi_target_forest.predict(valid_ft_mat)

        test_pred = multi_target_forest.predict(test_ft_mat)

        print('train: ', evaluate_regression(train_pred, train_label_mat))
        print('valid: ', evaluate_regression(valid_pred, valid_label_mat))
        print('test:  ', evaluate_regression(test_pred, test_label_mat))

        # return test mse mape
        mse, mae, mape = evaluate_regression(test_pred, test_label_mat)
        return mse, mae, mape

    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)


if __name__ == '__main__':
    mse_res_list = []
    mae_res_list = []
    mape_res_list = []
    for pred_step in [1, 3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = KNNTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type).start()
                # res_list.append(res[2])  # mse
                mse_res_list.append(res[0])  # mse
                mae_res_list.append(res[1])  # mae
                mape_res_list.append(res[2])  # mape

    print(f'MSE: {[mse_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAE: {[mae_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAPE: {[mape_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')









