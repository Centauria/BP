# -*- coding: utf-8 -*-
import abc


class Forward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, input_value):
        """
        Forward calculation.
        :param input_value:
        :return:
        """
        pass


class Backward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self, output_value):
        """
        Backward calculation.
        :param output_value:
        :return:
        """
        pass


class Neuron:
    def __init__(self):
        self._input = []
        self._weight = []
        self._output = []
