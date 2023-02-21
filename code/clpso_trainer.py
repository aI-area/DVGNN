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

from tools.evaluate_utils import evaluate_regression
from dataset.load_dataset import GlobalFlu
from sklearn.metrics.pairwise import pairwise_kernels


class MSVR():
    def __init__(self, kernel='rbf', degree=3, gamma=None, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
        super(MSVR, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.Beta = None
        self.NSV = None
        self.xTrain = None

    def fit(self, x, y):
        self.xTrain = x.copy()
        C = self.C
        epsi = self.epsilon
        tol = self.tol

        n_m = np.shape(x)[0]  # num of samples
        n_d = np.shape(x)[1]  # input data dimensionality
        n_k = np.shape(y)[1]  # output data dimensionality (output variables)

        # H = kernelmatrix(ker, x, x, par)
        H = pairwise_kernels(x, x, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)

        self.Beta = np.zeros((n_m, n_k))

        #E = prediction error per output (n_m * n_k)
        E = y - np.dot(H, self.Beta)
        #RSE
        u = np.sqrt(np.sum(E**2, 1, keepdims=True))

        #RMSE
        RMSE = []
        RMSE_0 = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_0)

        #points for which prediction error is larger than epsilon
        i1 = np.where(u > epsi)[0]

        #set initial values of alphas a (n_m * 1)
        a = 2 * C * (u - epsi) / u

        #L (n_m * 1)
        L = np.zeros(u.shape)

        # we modify only entries for which  u > epsi. with the sq slack
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

        #Lp is the quantity to minimize (sq norm of parameters + slacks)
        Lp = []
        BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
        Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_0)

        eta = 1
        k = 1
        hacer = 1
        val = 1

        while(hacer):
            Beta_a = self.Beta.copy()
            E_a = E.copy()
            u_a = u.copy()
            i1_a = i1.copy()

            M1 = H[i1][:, i1] + \
                np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))

            #compute betas
            sal1 = np.dot(np.linalg.inv(M1), y[i1])

            eta = 1
            self.Beta = np.zeros(self.Beta.shape)
            self.Beta[i1] = sal1.copy()

            #error
            E = y - np.dot(H, self.Beta)
            #RSE
            u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)
            i1 = np.where(u >= epsi)[0]

            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

            #%recompute the loss function
            BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp.append(Lp_k)

            #Loop where we keep alphas and modify betas
            while(Lp[k] > Lp[k-1]):
                eta = eta/10
                i1 = i1_a.copy()

                self.Beta = np.zeros(self.Beta.shape)
                #the new betas are a combination of the current (sal1)
                #and of the previous iteration (Beta_a)
                self.Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]

                E = y - np.dot(H, self.Beta)
                u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)

                i1 = np.where(u >= epsi)[0]

                L = np.zeros(u.shape)
                L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
                BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
                Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
                Lp[k] = Lp_k

                #stopping criterion 1
                if(eta < 1e-16):
                    Lp[k] = Lp[k-1] - 1e-15
                    self.Beta = Beta_a.copy()

                    u = u_a.copy()
                    i1 = i1_a.copy()

                    hacer = 0

            #here we modify the alphas and keep betas
            a_a = a.copy()
            a = 2 * C * (u - epsi) / u

            RMSE_k = np.sqrt(np.mean(u**2))
            RMSE.append(RMSE_k)

            if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
                hacer = 0

            k = k + 1

            #stopping criterion #algorithm does not converge. (val = -1)
            if(len(i1) == 0):
                hacer = 0
                self.Beta = np.zeros(self.Beta.shape)
                val = -1

        self.NSV = len(i1)

    def predict(self, x):
        H = pairwise_kernels(x, self.xTrain, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        yPred = np.dot(H, self.Beta)
        return yPred


class CLPSOTrainer(object):
    def __init__(self, data_type='us', split_param=[0.6, 0.2, 0.2], wind_size=12, pred_step=1, seed=3):
        self.setup_seed(seed)
        print('CLPSO_ML')

        self.wind_size = wind_size
        self.pred_step = pred_step

        # model init params
        self.param_range = {'C': np.logspace(0, 2, 16), 'epsilon': np.logspace(-4, -2, 4), 'gamma': [0.05, 0.1, 0.2, 0.4]}
        self.param_keys = ['C', 'epsilon', 'gamma']

        # CLPSO params
        self.maxIter = 200
        self.maxFailIter = 30
        self.C = 2
        self.w_max = 0.9
        self.w_min = 0.4
        self.N = 8
        self.D = wind_size + self.model_param_to_particle()[0]
        self.M = 8  # refreshing gap

        self.dataset = GlobalFlu(data_type=data_type, split_param=split_param, wind_size=wind_size,
                                 pred_step=pred_step)

    def model_param_to_particle(self):
        D_param_part = 0
        bin_length = {}

        for p_n in self.param_keys:
            bin_length[p_n] = np.log2(len(self.param_range[p_n])).astype(int)
            D_param_part += bin_length[p_n]

        return D_param_part, bin_length

    def particle_to_modelParam(self, p_i):
        bin_length = self.model_param_to_particle()[1]
        param_bin = {}
        param_restore = {}

        index_start = 0
        for p_n in self.param_keys:
            param_bin[p_n] = p_i[index_start:(index_start + bin_length[p_n])].copy()
            index_start = index_start + bin_length[p_n]
            param_restore[p_n] = self.param_range[p_n][param_bin[p_n].dot(2 ** np.arange(param_bin[p_n].size)[::-1])]

        params_dict = param_restore.copy()
        params_dict['kernel'] = "rbf"

        return params_dict, list(param_restore.values())

    def particle_to_features(self, p_i):
        if p_i[-self.wind_size:].all() == 0:
            p_i[-1] = 1

        feature_mask = p_i[-self.wind_size:].copy()

        return feature_mask, p_i

    def training_model(self, X_train, y_train, X_valid, y_valid, feature_mask, params_dict):
        # feature
        feature_index = np.arange(self.wind_size)
        selected_feature_index = feature_index[feature_mask == 1]
        selected_X_train = X_train[0:X_train.shape[0], feature_mask == 1]
        selected_X_valid = X_valid[0:X_valid.shape[0], feature_mask == 1]

        # model and parameters
        model = MSVR(**params_dict)
        model.fit(selected_X_train, y_train)
        y_pred = model.predict(selected_X_valid)
        mse = evaluate_regression(y_pred, y_valid)[0]

        return mse, model, selected_feature_index

    def run_clpso(self, X_train, y_train, X_valid, y_valid):
        # initialization p, v, pbest
        p = np.random.randint(0, 2, (self.N, self.D))
        v = np.zeros((self.N, self.D))
        pbest = p.copy()

        t = 0
        t_useless = 0

        fit_p = np.zeros((self.N, 1))
        fit_pbest = np.zeros((self.N, 1))

        refresh_gap = np.full((self.N, 1), self.M + 1)
        exemplar = np.zeros((self.N, self.D))

        s_v = np.zeros((self.N, self.D))  # sigmoid function of v

        pc = np.zeros((self.N, 1))  # predefined probability(Pc(i))
        for i in range(self.N):
            pc[i] = 0.05 + 0.45 * (np.exp(10 * (i - 1) / (self.N - 1)) - 1) / (np.exp(10) - 1)

        fit_gbest_all = []

        # evaluate the swarm and initialize the gbest
        for i in range(self.N):
            # train model
            feature_mask, p[i] = self.particle_to_features(p[i])
            params_dict = self.particle_to_modelParam(p[i])[0]
            fit_p[i] = self.training_model(X_train, y_train, X_valid, y_valid, feature_mask, params_dict)[0]

            fit_pbest[i] = fit_p[i].copy()

        fit_gbest = fit_pbest.min().copy()
        gbest = pbest[fit_pbest.argmin()].copy()

        while (t < self.maxIter and t_useless < self.maxFailIter):

            w = ((self.w_max - self.w_min) * (self.maxIter - t)) / self.maxIter + self.w_min
            p_not_updated = list(range(self.N))

            for i in range(self.N):
                # assign exemplar for each d
                if refresh_gap[i] > self.M:
                    if (len(p_not_updated) > 1):
                        p_candidate = list(p_not_updated)
                        p_candidate.remove(i)
                    else:
                        p_candidate = []

                    for d in range(self.D):
                        if random.random() > pc[i]:
                            exemplar[i][d] = i
                        else:
                            if (p_candidate != []):

                                p1_i = random.choice(p_candidate)

                                p_candidate2 = list(p_candidate).copy()
                                if len(p_candidate) > 1:
                                    p_candidate2.remove(p1_i)
                                p2_i = random.choice(p_candidate2)

                                exemplar[i][d] = p1_i if fit_pbest[p1_i] < fit_pbest[p2_i] else p2_i
                            else:
                                exemplar[i][d] = i
                    if exemplar[i].all() == i:
                        exemplar[i][random.randint(0, self.D - 1)] = random.randint(0, self.N - 1)

                exemplar_int = exemplar.astype(int)

                # update v,p
                for d in range(self.D):
                    v[i][d] = w * v[i][d] + self.C * random.random() * (pbest[exemplar_int[i][d]][d] - p[i][d])
                    s_v[i][d] = 1 / (1 + np.exp(-v[i][d]))
                    p[i][d] = 1 if random.random() < s_v[i][d] else 0

                p_not_updated.remove(i)

                # calculate fitness function and update pbest,fit_pbest
                feature_mask, p[i] = self.particle_to_features(p[i])
                params_dict = self.particle_to_modelParam(p[i], )[0]
                fit_p[i] = self.training_model(X_train, y_train, X_valid, y_valid, feature_mask, params_dict)[0]

                if fit_p[i] < fit_pbest[i]:
                    pbest[i] = p[i].copy()
                    fit_pbest[i] = fit_p[i].copy()
                    refresh_gap[i] = 0
                else:
                    refresh_gap[i] += 1

                    # update gbest,fit_gbest
            if fit_pbest.min() < fit_gbest:

                gbest = pbest[fit_pbest.argmin()].copy()

                fit_gbest = fit_pbest.min().copy()

                t_useless = 0
            else:

                t_useless += 1

            fit_gbest_all.append(fit_gbest)

            t += 1

        feature_mask, gbest = self.particle_to_features(gbest)
        params_dict, best_params = self.particle_to_modelParam(gbest)
        fit_gbest, best_model, selected_feature_index = self.training_model(X_train, y_train, X_valid, y_valid, feature_mask, params_dict)

        return fit_gbest, best_model, selected_feature_index, best_params

    def predict(self, X, model, selected_feature_index):
        selected_X = X[:, selected_feature_index]
        return model.predict(selected_X)

    def start(self):

        train_ft_mat = self.dataset.ft_mat[self.dataset.train_index]
        train_label_mat = self.dataset.label_mat[self.dataset.train_index]

        valid_ft_mat = self.dataset.ft_mat[self.dataset.valid_index]
        valid_label_mat = self.dataset.label_mat[self.dataset.valid_index]

        test_ft_mat = self.dataset.ft_mat[self.dataset.test_index]
        test_label_mat = self.dataset.label_mat[self.dataset.test_index]

        # train model
        fit_gbest, best_model, selected_feature_index, best_params = self.run_clpso(train_ft_mat, train_label_mat, valid_ft_mat, valid_label_mat)

        # predict
        train_pred = self.predict(train_ft_mat, best_model, selected_feature_index)
        valid_pred = self.predict(valid_ft_mat, best_model, selected_feature_index)
        test_pred = self.predict(test_ft_mat, best_model, selected_feature_index)

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
                res = CLPSOTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type).start()
                # res_list.append(res[2])  # mse
                mse_res_list.append(res[0])  # mse
                mae_res_list.append(res[1])  # mae
                mape_res_list.append(res[2])  # mape

    print(f'MSE: {[mse_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAE: {[mae_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')
    print(f'MAPE: {[mape_res_list[i] for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]]}')









