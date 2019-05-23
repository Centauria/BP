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
    def __init__(self, name='', bias: Optional[np.float] = None, activate_function: Function = Function.sigmoid()):
        # input_list_values: the links that connect to this neuron
        self.__input_list = []
        # output: the links that this neuron connects to
        self.__output_list = []
        self.__f: Function = activate_function
        self.name = name
        # short-circuit variables
        self.__bias: np.float = bias if bias is not None else np.random.rand()
        self.__target: Optional[np.float] = None
    
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
        return np.array(result, dtype=np.float)
    
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
        return np.array(result, dtype=np.float)
    
    @property
    def bias(self) -> np.float:
        return self.__bias
    
    @bias.setter
    def bias(self, bias: np.float):
        self.__bias = bias
    
    @property
    def output_list(self) -> list:
        return self.__output_list
    
    @property
    def input_list(self) -> list:
        return self.__input_list
    
    @property
    def target(self) -> np.float:
        return self.__target
    
    @target.setter
    def target(self, target: Optional[np.float]):
        self.__target = target
    
    @property
    def activate_function(self):
        return self.__f
    
    @activate_function.setter
    def activate_function(self, f):
        self.__f = f
    
    def forward(self) -> np.float:
        result = self.__f(self.__bias + np.sum(self.input_list_values))
        return result
    
    def backward(self) -> np.float:
        v = self.__bias + np.sum(self.input_list_values)
        if self.__target is None:
            delta_result = self.__f.d(v) * np.sum([l.backward() for l in self.__output_list])
        else:
            delta_result = self.__f.d(v) * (self.__target - self.forward())
        return delta_result
    
    def connect(self, other, weight=None):
        return connect(self, other, weight)
    
    def commit(self, ita):
        self.__bias += ita * self.backward()
        for link in self.__output_list:
            link.commit(ita)
        self.__target = None


class Link(Forward, Backward):
    def __init__(self, source: Optional[Neuron], destination: Optional[Neuron], weight=0.0):
        self.source = source
        self.destination = destination
        self.weight = weight
    
    def __repr__(self):
        return '%s <--%.10f--> %s' % (self.source, self.weight, self.destination)
    
    def forward(self):
        return self.source.forward() * self.weight
    
    def backward(self):
        return self.destination.backward() * self.weight
    
    def commit(self, ita):
        self.weight += ita * self.destination.backward() * self.source.forward()
        # self.destination.commit(ita)


def connect(neuron_1: Neuron, neuron_2: Neuron, weight: Optional[np.float] = None) -> Link:
    weight = weight if weight is not None else np.random.rand()
    link = Link(neuron_1, neuron_2, weight)
    neuron_1.output_list.append(link)
    neuron_2.input_list.append(link)
    return link
