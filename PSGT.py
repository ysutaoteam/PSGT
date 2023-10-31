#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from itertools import chain
from tqdm import tqdm
import math


def ParkinsonLoader(path1):
    parkinson_x = pd.read_csv(path1)
    columns = ['Unnamed: 0', ]
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)
    columns = ['motor_UPDRS', 'total_UPDRS']
    parkinson_x = parkinson_x.drop(columns, axis=1)

    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')

    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(parkinson_x, parkinson_y, test_size=0.4, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
    return x_train, x_val, x_test, y_train, y_val, y_test


def train(trainx, testx, trainy, testy):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(trainx)
    x_test = scaler.transform(testx)
    scale1 = MinMaxScaler()
    y_train = scale1.fit_transform(trainy)
    y_test = scale1.transform(testy)
    y_train = y_train.reshape(y_train.shape[0], )

    from sklearn import datasets, ensemble
    rf_model = ensemble.RandomForestRegressor(n_estimators=30, max_depth=50, criterion='mse')
    rf_model.fit(x_train, y_train)
    pre = rf_model.predict(x_test)

    dataset_pred = pre.reshape((-1, 1))
    dataset_pred = np.array(dataset_pred).reshape((-1, 1))

    dataset_pred = scale1.inverse_transform(dataset_pred)
    y_test = scale1.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test, dataset_pred))
    mae = mean_absolute_error(y_test, dataset_pred)
    r2 = r2_score(y_test, dataset_pred)

    def volatility(b, a):  # b:真实值  a:预测值
        c = b - a
        mean1 = np.mean(c)
        v = np.mean(np.abs(c - mean1))
        return v

    vol = volatility(y_test, dataset_pred)

    return mae, rmse, r2, vol



def res(x1):
    parkinson_x = pd.read_csv(x1)
    columns = ['Unnamed: 0', ]
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)
    columns = ['motor_UPDRS', 'total_UPDRS']
    parkinson_x = parkinson_x.drop(columns, axis=1)

    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')

    c = np.hstack((parkinson_x, parkinson_y))
    np.random.shuffle(c)
    parkinson_x = np.array(c[:, :-1])
    parkinson_y = np.array(c[:, -1]).reshape((-1, 1))
    return parkinson_x, parkinson_y


def w(money, zuhe, n):

    shap = [0 for x in range(n)]
    for j in range(n):
        for i in zuhe:
            if i == [j]:
                s = zuhe
                ss = []
                for k in s:
                    if j not in k:
                        ss.append(k)
                for k in ss:
                    weight = math.factorial(len(list(k))) * math.factorial(n - len(list(k)) - 1) / math.factorial(n)
                    r = list(np.hstack((list(k), list([j]))))
                    r = sorted(r)
                    b = ''.join(str(r))
                    e = ''.join(str(k))
                    contrib = money[b] - money[e]
                    shap[j] += weight * contrib
        e = ''.join(str([j]))
        shap[j] += math.factorial(n - 1) * money[e] / math.factorial(n)

    shap = [-i for i in shap]
    a = [1 / (1 + np.exp(-i)) for i in shap]
    shap1 = a / np.sum(a)
    return shap1


def shap(money, zuhe):
    P = set(chain(*zuhe))
    def len1(a):
        int_count = 0
        for i in a:
            if i.isdigit():
                int_count += 1
        return int_count

    def phi(channel_index):
        S_channel = [k for k in money.keys() if str(channel_index) in k]
        score = 0
        # print(f"Computing phi for channel {channel_index}...")
        for S in tqdm(S_channel):
            score += money[S] / len1(S)
        return score
    value1 = []
    for j in P:
        value = phi(j)
        value1.append(value)
    return value1

all_sum = [] 
all_sum_rmse = []  
all_sum_r2 = []  
all_sum_vol = []  
for kk in range(42):
    p111 = "p"
    p211 = ".csv"
    c11 = np.hstack((p111, kk + 1))
    c11 = np.hstack((c11, p211))
    s11 = ""
    csv = s11.join(c11)

    tx_train, tx_val, tx_test, ty_train, ty_val, ty_test = ParkinsonLoader(csv)

    p11 = "motor-mae"
    p21 = ".csv"
    c1 = np.hstack((p11, kk + 1))
    c1 = np.hstack((c1, p21))
    s1 = ""
    csv = s1.join(c1)

    parkinson_x = pd.read_csv(csv)
    columns = ['Unnamed: 0']
    sort = parkinson_x.drop(columns, axis=1)
    dataset = sort.values
    sort = dataset.astype('float32')

    from itertools import compress, product
    def combinations(items):
        return (list(compress(items, mask)) for mask in product(*[[0, 1]] * len(items)))

    n = 5
    weight = []
    zuhe = list(combinations(range(n)))
    zuhe.remove(zuhe[0])

    mae111, rmse, r2, vol = train(tx_train, tx_val, ty_train, ty_val)
    money = {}
    for i in zuhe:
        if len(i) == 1:
            for j in i:
                c = np.hstack(("p", int(sort[j, 0])))
                c = np.hstack((c, ".csv"))
                b00 = "".join(c)
                xtrain, ytrain = res(b00)

                xtrain = np.vstack((tx_train, xtrain))
                ytrain = np.vstack((ty_train, ytrain))
                mae, rmse, r2, vol = train(xtrain, tx_val, ytrain, ty_val)
                money[str(i)] = np.abs(mae111 - mae)
        if len(i) >= 2:
            x = []
            y = []
            x.extend(tx_train)
            y.extend(ty_train)
            for j in i:
                c = np.hstack(("p", int(sort[j, 0])))
                c = np.hstack((c, ".csv"))
                b00 = "".join(c)
                xtrain, ytrain = res(b00)
                x.extend(xtrain)
                y.extend(ytrain)
            np.random.shuffle(x)
            x = np.array(x).flatten().reshape((-1, tx_test.shape[1]))
            y = np.array(y).reshape((-1, 1))
            mae, rmse, r2, vol = train(x, tx_val, y, ty_val)

            money[str(i)] = np.abs(mae111 - mae)

    weight = w(money, zuhe, n)  
    kweight.append(weight)

    tx_train1 = tx_train
    ty_train1 = ty_train
    for j in range(n):
        c = np.hstack(("p", int(sort[j, 0])))
        c = np.hstack((c, ".csv"))
        b00 = "".join(c)
        xtrain, ytrain = res(b00)
        import networkx as nx
        G = nx.random_graphs.erdos_renyi_graph(xtrain.shape[0], 0.2)  # 生成包含1000个节点、连边概率0.6的随机图
        c = nx.find_cliques(G)
        
        mae1, rmse, r2, vol = train(tx_train, tx_val, ty_train, ty_val)
        money1 = {}
        li = []
        for i in c:
            x_train = np.vstack((tx_train, xtrain[i]))
            y_train = np.vstack((ty_train, ytrain[i]))
            mae2, rmse, r2, vol = train(x_train, tx_val, y_train, ty_val)
            money1[str(i)] = np.abs(mae1 - mae2)
            li.append(i)

        shaply = shap(money1, li)
        zheng = 0
        fu = 0
        for i in shaply:
            if i > 0:
                zheng += 1
            if i <= 0:
                fu += 1
        ratio_zheng = zheng / len(shaply)
        ratio_fu = fu / len(shaply)

        ratio.append(ratio_zheng)
        ratio.append(ratio_fu)
        shaply = [-i for i in shaply]
        shaply = sorted(range(len(shaply)), key=lambda k: shaply[k], reverse=True)
        w1 = shaply[0:int(len(shaply) * weight[j] / 3)]
        tx_train1 = np.vstack((tx_train1, xtrain[w1]))
        ty_train1 = np.vstack((ty_train1, ytrain[w1]))

    tx_mae, tx_rmse, tx_r2, tx_vol = train(tx_train1, tx_test, ty_train1, ty_test)
    ori_mae1, ori_rmse, ori_r2, ori_vol = train(tx_train, tx_test, ty_train, ty_test)

    all_sum.append(tx_mae)
    all_sum_rmse.append(tx_rmse)
    all_sum_r2.append(tx_r2)
    all_sum_vol.append(tx_vol)

def error(all_sum):
    t_mae = np.mean(np.array(all_sum))
    t_mae_var = np.var(np.array(all_sum))
    t_mae_std = np.std(np.array(all_sum))
    return t_mae,t_mae_var,t_mae_std

t_mae,t_mae_var,t_mae_std=error(all_sum)
t_rmse,t_rmse_var,t_rmse_std=error(all_sum_rmse)
t_vol,t_vol_var,t_vol_std=error(all_sum_vol)


