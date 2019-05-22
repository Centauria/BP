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
    def backward(self):
        """
        Backward calculation.
        :return:
        """
        pass


class Neuron(Forward, Backward):
    def __init__(self, name='', activate_function: Function = Function.sigmoid()):
        """input: the links that connect to this neuron"""
        self.__input = []
        """output: the links that this neuron connects to"""
        self.__output = []
        self.__f = activate_function
        self.name = name
        self.__delta = 0
    
    @property
    def weight(self) -> np.ndarray:
        result = []
        for link in self.__input:
            result.append(link.weight)
        return np.array(result, dtype=np.float32)
    
    @weight.setter
    def weight(self, weight: iter):
        assert len(weight) == len(self.__input), 'vector length not match'
        for i in range(len(weight)):
            self.__input[i].weight = weight[i]
    
    @property
    def input(self) -> np.ndarray:
        result = []
        for link in self.__input:
            result.append(link.forward())
        return np.array(result, dtype=np.float32)
    
    @property
    def output_list(self) -> list:
        return self.__output
    
    @property
    def input_list(self) -> list:
        return self.__input
    
    def forward(self):
        return np.array(self.__f(np.dot(self.weight, self.input))).flatten()
    
    def backward(self, delta=None):
        if delta is None:
            v_j = np.dot(self.weight, self.input)
            delta_result = self.__delta + \
                           np.array(self.__f.d(v_j)) * np.sum([link.backward() for link in self.__output])
            self.__delta = np.zeros((1,))
        else:
            self.__delta = np.array(delta).flatten()
            delta_result = self.__delta
        return delta_result
    
    def connect(self, other, weight=0.0):
        return connect(self, other, weight)
    
    def __repr__(self):
        return '<%s>' % self.name


class Link(Forward, Backward):
    def __init__(self, source: Neuron, destination: Neuron, weight=0.0):
        self.source = source
        self.destination = destination
        self.weight = weight
    
    def forward(self):
        return self.source.forward() * self.weight
    
    def backward(self):
        return self.destination.backward() * self.weight
    
    def commit(self, ita):
        self.weight += ita * self.destination.backward() * self.source.forward()
    
    def __repr__(self):
        return '%s <--%.3f--> %s' % (self.source, self.weight, self.destination)


def connect(neuron_1: Neuron, neuron_2: Neuron, weight=0.0) -> Link:
    link = Link(neuron_1, neuron_2, weight)
    neuron_1.output_list.append(link)
    neuron_2.input_list.append(link)
    return link
