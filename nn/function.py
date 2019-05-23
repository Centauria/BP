# -*- coding: utf-8 -*-
import numpy as np


class Function:
    def __init__(self, func, derivative):
        self.__f = func
        self.__d = derivative
        pass
    
    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            result = self.__f(args[0])
        elif len(args) > 1:
            result = []
            for arg in args:
                result.append(self.__f(arg))
        else:
            raise ValueError('No args found')
        return result
    
    def d(self, *args):
        if len(args) == 1:
            result = self.__d(args[0])
        elif len(args) > 1:
            result = []
            for arg in args:
                result.append(self.__d(arg))
        else:
            raise ValueError('No args found')
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
        relud = lambda x: 1 if x >= 0 else 0
        return Function(relu, relud)
    
    @staticmethod
    def bias(level=1.0):
        one = lambda x: level
        zero = lambda x: 0
        return Function(one, zero)
