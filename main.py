import pandas as pd
import numpy as np
import datetime as dt
from gurobipy import *
from matplotlib import pyplot as plt
from copy import *
from collections import *

class utils():
    ## generate model parameters
    def generate_params(sample_skus, sample_dcs_dummy, market_share_mean, cov, price_over_skus, total_attraction, profit_ratio, seed):
        np.random.seed(seed)
        cost, revenue, attraction = {}, {}, {}

        flex = [(3, 14), (3, 33), (3, 34), (3, 35), (3, 64)]
        for k in sample_skus:
            ## fulfillment cost
            for j in sample_dcs_dummy[1:]:
                for i in sample_dcs_dummy:
                    if i == j:
                        cost[k, i, j] = np.random.normal(2, 0)
                    elif (i, j) in flex:
                        cost[k, i, j] = np.random.normal(4, 0)
                    elif i == sample_dcs_dummy[0]:
                        cost[k, i, j] = np.random.normal(8, 0)
                    else:
                        cost[k, i, j] = np.random.normal(1000, 0)

            revenue[k] = price_over_skus[k] * profit_ratio

            ## revenue
            mean_attraction = market_share_mean[sample_skus.index(k)] * total_attraction
            for j in sample_dcs_dummy[1:]:
                attraction[k, j] = max(0.0001, np.random.normal(mean_attraction, mean_attraction * cov))

        return cost, revenue, attraction

    ## projection to L-1 ball
    def projection_l1(x, q):
        if sum(x) > q:
            mu = sorted(x, reverse=True)
            largest_idx = 0
            for l in range(len(mu)):
                if mu[l] - (sum(mu[:l+1]) - q) / (l+1) > 0:
                    largest_idx = l
                else:
                    break
            theta = (sum(mu[:largest_idx+1]) - q) / (largest_idx+1)
            for l in range(len(mu)):
                x[l] = max(x[l] - theta, 0)

        return x

class optimizers():
    def MNL_det(T, sample_skus, sample_dcs_dummy, cost, revenue, attraction, dc_freq, card, capa, S_init=None):
        model = Model()
        model.setParam("LogToConsole", 0)

        X = model.addVars(((k, i, j) for k in sample_skus for i in sample_dcs_dummy for j in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='fulfill')
        omega = model.addVars(((k, j) for k in sample_skus for j in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='assort')
        omega_outside = model.addVars((j for j in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='assort_offset')
        max_omega = model.addVars((k for k in sample_skus), vtype=GRB.BINARY, name='max_assort')
        S = model.addVars(((k, i) for k in sample_skus for i in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='size')
        model.setObjective(sum(X[k, i, j] * (revenue[k] - cost[k, i, j]) for k, i, j in X), GRB.MAXIMIZE)

        ## fulfillment cost
        for k in sample_skus:
            for i in sample_dcs_dummy[1:]:
                model.addConstr(X.sum(k, i, '*') <= S[k, i], name='supply_%s_%s' % (k, i))

            for j in sample_dcs_dummy[1:]:
                model.addConstr(X.sum(k, '*', j) <= T * dc_freq[sample_dcs_dummy.index(j)-1] * omega[k, j], name='demand_%s_%s' % (k, j))

        for i in sample_dcs_dummy[1:]:
            model.addConstr(S.sum('*', i) <= capa[i])

        ## revenue assortment
        for j in sample_dcs_dummy[1:]:
            model.addConstr(omega.sum('*', j) + omega_outside[j] == 1)

            for k in sample_skus:
                model.addConstr(omega[k, j] / attraction[k, j] <= omega_outside[j])
                model.addConstr(omega[k, j] / attraction[k, j] <= max_omega[k])

        #     model.addConstr(omega_outside[j] >= min_omega_outside)
        # model.addConstr(sum(max_omega[k] for k in sample_skus) <= card * min_omega_outside)
        model.addConstr(sum(max_omega[k] for k in sample_skus) <= card)

        # j = sample_dcs[0]
        # model.addConstr(sum(omega[k, j] / attraction[k, j] for k in sample_skus) <= card * omega_outside[j])

        if S_init is not None:
            for k in sample_skus:
                for i in sample_dcs_dummy[1:]:
                    model.addConstr(S[k, i] == S_init[k, i])

        model.optimize()
        obj = model.objVal

        return model, X, omega, omega_outside, S, obj

    def MNL_iso(T, sample_skus, sample_dcs_dummy, cost, revenue, attraction, dc_freq, card, capa, S_init=None):
        model = Model()
        model.setParam("LogToConsole", 0)
        model.setParam("TIME_LIMIT", 600)

        omega = model.addVars(((k, j) for k in sample_skus for j in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='assort')
        omega_outside = model.addVars((j for j in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='assort_offset')
        max_omega = model.addVars((k for k in sample_skus), vtype=GRB.BINARY, name='max_assort')
        model.setObjective(sum((revenue[k]-0) * dc_freq[sample_dcs_dummy.index(j)-1] * omega[k, j] for k, j in omega), GRB.MAXIMIZE)

        ## revenue assortment
        for j in sample_dcs_dummy[1:]:
            model.addConstr(omega.sum('*', j) + omega_outside[j] == 1)

            for k in sample_skus:
                model.addConstr(omega[k, j] / attraction[k, j] <= omega_outside[j])
                model.addConstr(omega[k, j] / attraction[k, j] <= max_omega[k])

        model.addConstr(sum(max_omega[k] for k in sample_skus) <= card)

        model.optimize()
        obj = model.objVal

        return model, omega, omega_outside, obj

    def sizing(T, sample_skus, sample_dcs_dummy, cost, revenue, capa, D):
        model = Model()
        model.setParam("LogToConsole", 0)

        X = model.addVars(((k, i, j, n) for k in sample_skus for i in sample_dcs_dummy for j in sample_dcs_dummy[1:] for n in range(len(D))), vtype=GRB.CONTINUOUS, name='fulfill')
        S = model.addVars(((k, i) for k in sample_skus for i in sample_dcs_dummy[1:]), vtype=GRB.CONTINUOUS, name='size')

        model.setObjective(sum(X[k, i, j, n] * (revenue[k] - cost[k, i, j]) for k, i, j, n in X), GRB.MAXIMIZE)

        for n in range(len(D)):
            for k in sample_skus:
                for i in sample_dcs_dummy[1:]:
                    model.addConstr(X.sum(k, i, '*', n) <= S[k, i], name='supply_%s_%s' % (k, i))

                for j in sample_dcs_dummy[1:]:
                    model.addConstr(X.sum(k, '*', j, n) <= D[n][k, j], name='demand_%s_%s' % (k, j))

        for i in sample_dcs_dummy[1:]:
            model.addConstr(S.sum('*', i) <= capa[i])

        model.optimize()
        obj = model.objVal

        return model, X, S, obj

def preprocessing():
    sample_orders = pd.read_csv('../sample_orders_top100.csv')
    #network = pd.read_csv('../JD_network_data.csv')

    demand_over_skus = sample_orders.groupby('sku_ID').count()['order_ID'].to_dict()
    sample_skus = [i for i, j in sorted(demand_over_skus.items(), key = lambda kv: kv[1], reverse=True)]

    demand_over_dcs = sample_orders.groupby('dc_des').count()['order_ID'].to_dict()
    sample_dcs = list(demand_over_dcs.keys())

    demand_over_skus_dcs = {}
    for i in sample_dcs:
        sample_orders_dc = sample_orders[sample_orders['dc_des'] == i]
        tmp = sample_orders_dc.groupby('sku_ID').count()['order_ID'].to_dict()
        demand_over_skus_dcs[i] = tmp

    market_share_dcs = []
    for i in sample_dcs:
        tmp = []
        for k in sample_skus:
            if k in demand_over_skus_dcs[i]:
                tmp.append(demand_over_skus_dcs[i][k] / demand_over_dcs[i])
            else:
                tmp.append(0)
        market_share_dcs.append(tmp)

    market_share_skus = []
    for k in sample_skus:
        tmp = []
        for i in sample_dcs:
            if k in demand_over_skus_dcs[i]:
                tmp.append(demand_over_skus_dcs[i][k] / demand_over_dcs[i])
            else:
                tmp.append(0)
        market_share_skus.append(tmp)

    market_share_mean = np.array(sorted(demand_over_skus.values(), reverse=True)) / sum(demand_over_skus.values())
    location_share_mean = [demand_over_dcs[i] / sum(demand_over_dcs.values()) for i in sample_dcs]

    cov = np.mean(np.std(np.array(market_share_skus), axis=1) / market_share_mean)

    price_over_skus = sample_orders.groupby('sku_ID').mean()['original_unit_price'].to_dict()

    return sample_dcs, sample_skus, location_share_mean, market_share_mean, cov, price_over_skus

def synthetic():
    sample_dcs = ['A', 'B', 'C', 'D']
    sample_skus = [k for k in range(1, 41)]
    location_share_mean = [0.25, 0.25, 0.25, 0.25]

    return sample_dcs, sample_skus, location_share_mean

def main():
    #sample_dcs, sample_skus, location_share_mean = synthetic()
    sample_dcs, sample_skus, location_share_mean, market_share_mean, cov, price_over_skus = preprocessing()
    sample_dcs_dummy = [0] + sample_dcs

    total_attraction = 1
    max_share = total_attraction / (total_attraction + 1)
    T = 10000
    card = 40
    #q = 0.1
    #profit_ratio = 0.1

    for q in [0.1]:
        capa = {}
        for i in sample_dcs:
            if i == sample_dcs[0]:
                capa[i] = T * max_share * 2 * q
            else:
                capa[i] = T * max_share * q
        for profit_ratio in [0.2, 0.25]:
            print(profit_ratio)
            output_ub, output_algo = [], []
            for seed in range(10):
                cost, revenue, attraction = utils.generate_params(sample_skus, sample_dcs_dummy, market_share_mean, cov, price_over_skus, total_attraction, profit_ratio, seed)
                #model, X, omega, omega_outside, S, obj = optimizers.MNL_det(T, sample_skus, sample_dcs_dummy, cost, revenue, attraction, location_share_mean, card, capa)
                model, omega, omega_outside, obj = optimizers.MNL_iso(T, sample_skus, sample_dcs_dummy, cost, revenue, attraction, location_share_mean, card, capa)
                _omega = model.getAttr('X', omega)
                _omega_outside = model.getAttr('X', omega_outside)
                assortment = {}
                for j in sample_dcs:
                    products = []
                    for k in sample_skus:
                        y = _omega[k, j] / (attraction[k, j] * _omega_outside[j])
                        if y > 0.01:
                            products.append(sample_skus.index(k))
                    assortment[j] = products

                for j in sample_dcs:
                    print(assortment[j])

                print(obj)
                output_ub.append(obj)
                print('-----------------')

                market_share_fixed = {}
                for j in sample_dcs:
                    sku_freq = []
                    denominator = 1 + sum([attraction[k, j] for k in sample_skus if sample_skus.index(k) in assortment[j]])
                    for k in sample_skus:
                        if sample_skus.index(k) in assortment[j]:
                            sku_freq.append(attraction[k, j] / denominator)
                        else:
                            sku_freq.append(0)
                    market_share_fixed[j] = sku_freq

                N = 200
                D = []
                for n in range(N):
                    demand = defaultdict(int)
                    for t in range(T):
                        j = np.random.choice(sample_dcs, p = location_share_mean)
                        k = np.random.choice(['NA'] + sample_skus, p = [1 - sum(market_share_fixed[j])] + market_share_fixed[j])
                        demand[k, j] += 1
                    D.append(demand)

                model, X, S, obj = optimizers.sizing(T, sample_skus, sample_dcs_dummy, cost, revenue, capa, D)
                _S = model.getAttr('X', S)
                print(obj / N)
                output_algo.append(obj / N)
                print('-----------------')

            print(np.mean(output_ub), np.mean(output_algo))

if __name__ == '__main__':
    main()
