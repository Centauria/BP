# -*- coding: utf-8 -*-
import abc
import numpy as np
from nn.function import Function


class Forward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        """
        Forward calculation.
        :return:
        """
        pass


class Backward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self, error):
        """
        Backward calculation.
        :param error:
        :return:
        """
        pass


class Neuron(Forward, Backward):
    def __init__(self, activate_function: Function = Function.sigmoid()):
        self._input = []
        self._weight = []
        self._output = []
        self._f = activate_function

    def forward(self):
        return self._f(np.dot(self._weight, self._input))

    def backward(self, error):
        v_j = np.dot(self._weight, self._input)
        delta_j = self._f.d(v_j) * error
        # TODO: do the BP algorithm
