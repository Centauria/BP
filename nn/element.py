# -*- coding: utf-8 -*-
import abc
import numpy as np
from typing import Optional
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
        # input_list_values: the links that connect to this neuron
        self.__input_list = []
        # output: the links that this neuron connects to
        self.__output_list = []
        self.__f: Function = activate_function
        self.name = name
        # short-circuit variables
        self.__bias: np.float32 = 0.0
        self.__target: Optional[np.float32] = None

    def __repr__(self):
        return '(%.3f, [#%i, %.3f], %s, [#%i, %s], %.3f)' % (
            self.backward(),
            len(self.__input_list),
            self.__bias,
            self.name,
            len(self.__output_list),
            self.target,
            self.forward()
        )
    
    @property
    def weight(self) -> np.ndarray:
        result = []
        for link in self.__input_list:
            result.append(link.weight)
        return np.array(result, dtype=np.float32)
    
    @weight.setter
    def weight(self, weight: iter):
        assert len(weight) == len(self.__input_list), 'vector length not match'
        for i in range(len(weight)):
            self.__input_list[i].weight = weight[i]
    
    @property
    def input_list_values(self) -> np.ndarray:
        result = []
        for link in self.__input_list:
            result.append(link.forward())
        return np.array(result, dtype=np.float32)

    @property
    def bias(self) -> np.float32:
        return self.__bias

    @bias.setter
    def bias(self, bias: np.float32):
        self.__bias = bias
    
    @property
    def output_list(self) -> list:
        return self.__output_list
    
    @property
    def input_list(self) -> list:
        return self.__input_list

    @property
    def target(self) -> np.float32:
        return self.__target

    @target.setter
    def target(self, target: Optional[np.float32]):
        self.__target = target

    @property
    def activate_function(self):
        return self.__f

    @activate_function.setter
    def activate_function(self, f):
        self.__f = f

    def forward(self) -> np.float32:
        result = self.__f(self.__bias + np.sum(self.input_list_values))
        return result

    def backward(self) -> np.float32:
        if self.__target is None:
            v = np.sum(self.input_list_values)
            delta_result = self.__f.d(v) * np.sum([l.backward() for l in self.__output_list])
        else:
            delta_result = self.__target - self.forward()
        return delta_result

    def connect(self, other, weight=None):
        return connect(self, other, weight)

    def commit(self, ita):
        self.__bias += ita * self.backward()
        for link in self.__input_list:
            link.commit(ita)
        self.__target = None


class Link(Forward, Backward):
    def __init__(self, source: Optional[Neuron], destination: Optional[Neuron], weight=0.0):
        self.source = source
        self.destination = destination
        self.weight = weight
    
    def __repr__(self):
        return '%s <--%.3f--> %s' % (self.source, self.weight, self.destination)
    
    def forward(self):
        return self.source.forward() * self.weight
    
    def backward(self):
        return self.destination.backward() * self.weight
    
    def commit(self, ita):
        self.weight += ita * self.destination.backward() * self.source.forward()
        self.source.commit(ita)


def connect(neuron_1: Neuron, neuron_2: Neuron, weight=None) -> Link:
    weight = weight if weight is not None else np.random.rand() * 0.1
    link = Link(neuron_1, neuron_2, weight)
    neuron_1.output_list.append(link)
    neuron_2.input_list.append(link)
    return link
