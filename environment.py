import random
import numpy as np
import pandas as pd


d2p = [0.86,
       0.78,
       0.72,
       0.66,
       0.61,
       0.55,
       0.49,
       0.44,
       0.39,
       0.34
       ]


class Student(object):
    def __init__(self):
        pass

    def init(self, difficulty):
        pass

    def next_state(self, state, r, t, p):
        pass


class DHP(Student):
    def __init__(self):
        super().__init__()
        parameters = pd.read_csv('./parameters.csv', index_col=None)
        self.__ra = float(parameters['ra'].values[0])
        self.__rb = float(parameters['rb'].values[0])
        self.__rc = float(parameters['rc'].values[0])
        self.__rd = float(parameters['rd'].values[0])
        self.__fa = float(parameters['fa'].values[0])
        self.__fb = float(parameters['fb'].values[0])
        self.__fc = float(parameters['fc'].values[0])
        self.__fd = float(parameters['fd'].values[0])

    def init(self, d):
        p = d2p[d-1]
        t = 0
        if random.random() < p:
            r = 1
        else:
            r = 0
        h = self.cal_start_halflife(d, r)
        new_state, new_halflife = [h, d], h
        return r, t, p, new_state, new_halflife

    def next_state(self, state, r, t, p):
        h, d = state[0], state[1]
        p = np.exp2(- t / h)
        if r == 1:
            nh = self.cal_recall_halflife(d, h, p)
            nd = d
        else:
            nh = self.cal_forget_halflife(d, h, p)
            nd = min(d + 2,18)
            # nd = d
        return [nh, nd], nh

    def cal_start_halflife(self, d, r):
        init_halflife = - 1 / np.log2(max(0.925 - 0.05 * d, 0.025))
        if r == 0:
            return init_halflife
        else:
            return init_halflife * 10

    def cal_recall_halflife(self, d, halflife, p_recall):
        return halflife * (
                1 + np.exp(self.__ra) * np.power(d, self.__rb) * np.power(halflife, self.__rc) * np.power(
            1 - p_recall, self.__rd))

    def cal_forget_halflife(self, d, halflife, p_recall):
        return np.exp(self.__fa) * np.power(d, self.__fb) * np.power(halflife, self.__fc) * np.power(
            1 - p_recall, self.__fd)
