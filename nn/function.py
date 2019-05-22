# -*- coding: utf-8 -*-
import numpy as np


class Function:
    def __init__(self, function, derivative):
        self._f = function
        self._d = derivative
        pass

    def __call__(self, *args, **kwargs):
        result = []
        for arg in args:
            result.append(self._f(arg))
        return result

    def d(self, *args):
        result = []
        for arg in args:
            result.append(self._d(arg))
        return result

    @staticmethod
    def sigmoid():
        sig = lambda x: 1 / (1 + np.exp(-x))
        sigd = lambda x: sig(x) * (1 - sig(x))
        return Function(sig, sigd)

    @staticmethod
    def tanh():
        th = lambda x: np.tanh(x)
        thd = lambda x: 1 - th(x) ** 2
        return Function(th, thd)

    @staticmethod
    def ReLU():
        relu = lambda x: x if x > 0 else 0
        relud = lambda x: 1 if x > 0 else 0
        return Function(relu, relud)
