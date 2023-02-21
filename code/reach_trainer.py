#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/6/4 13:01

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import numpy as np
import random
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tools.evaluate_utils import evaluate_regression
from dataset.load_dataset import GlobalFlu
from sklearn.multioutput import MultiOutputRegressor

class REACHTrainer(object):
    def __init__(self, data_type='us', split_param=[0.6, 0.2, 0.2], wind_size=12, pred_step=1, seed=3, repeat=10):
        self.setup_seed(seed)

        print('REACH wind_size={} pred_step={}, data_type={}'.format(wind_size, pred_step, data_type))
        self.dataset = GlobalFlu(data_type=data_type, split_param=split_param, wind_size=wind_size, pred_step=pred_step)
        self.model_list = [[RandomForestRegressor() for _ in range(pred_step)] for _ in range(repeat)]
        self.order_list = [np.random.permutation(range(pred_step)) for _ in range(repeat)]

        self.pred_step = pred_step
        self.repeat = repeat

    def model_fit(self, ft_mat, label_mat):
        for i in range(self.repeat):
            tmp_label_mat = label_mat[:, self.order_list[i]]
            tmp_ft_mat = ft_mat

            self.model_list[i][0].fit(tmp_ft_mat, tmp_label_mat[:, 0])
            for j in range(1, self.pred_step):
                tmp_ft_mat = np.insert(tmp_ft_mat, tmp_ft_mat.shape[-1], tmp_label_mat[:,j-1], axis=-1)
                self.model_list[i][j].fit(tmp_ft_mat, tmp_label_mat[:,j])

    def model_predict(self, ft_mat):
        total_result = []
        for i in range(self.repeat):
            tmp_ft_mat = ft_mat[:,:]
            result = self.model_list[i][0].predict(tmp_ft_mat).reshape(-1, 1)
            for j in range(1, self.pred_step):
                tmp_ft_mat = np.insert(tmp_ft_mat, tmp_ft_mat.shape[-1], result[:, -1], axis=-1)
                pred = self.model_list[i][j].predict(tmp_ft_mat)
                result = np.insert(result, result.shape[-1], pred, axis=-1)

            recover_order = sorted(zip(self.order_list[i], range(self.pred_step)), key=lambda x: x[0])
            recover_order = [x[1] for x in recover_order]
            result = result[:, recover_order]
            total_result.append(result)

        total_result = np.stack(total_result, axis=-1)
        assemble_result = total_result.mean(axis=-1)

        return assemble_result



    def start(self):
        train_ft_mat = self.dataset.ft_mat[self.dataset.train_index]
        train_label_mat = self.dataset.label_mat[self.dataset.train_index]

        valid_ft_mat = self.dataset.ft_mat[self.dataset.valid_index]
        valid_label_mat = self.dataset.label_mat[self.dataset.valid_index]

        test_ft_mat = self.dataset.ft_mat[self.dataset.test_index]
        test_label_mat = self.dataset.label_mat[self.dataset.test_index]

        # model fit
        self.model_fit(train_ft_mat, train_label_mat)

        # model predict
        train_pred = self.model_predict(train_ft_mat)
        valid_pred = self.model_predict(valid_ft_mat)
        test_pred = self.model_predict(test_ft_mat)

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
                res = REACHTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type).start()
                mse_res_list.append(res[0])  # mse
                mae_res_list.append(res[1])  # mae
                mape_res_list.append(res[2])  # mape

    print(f'MSE: {[mse_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAE: {[mae_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAPE: {[mape_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
