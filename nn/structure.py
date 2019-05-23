# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, List
from nn.concept import Forward, Backward, Adaptable
from nn.element import Neuron
from nn.function import Function


class Layer(Forward, Backward, Adaptable):
    def __init__(self, length: int, name='', activate_function=Function.sigmoid()):
        self.name = name
        self.cells: List[Neuron] = [
            Neuron('_'.join((self.name, str(i))), activate_function=activate_function)
            for i in range(length)
        ]
    
    @property
    def bias(self) -> np.ndarray:
        return np.array([c.bias for c in self.cells])
    
    @bias.setter
    def bias(self, bias: np.ndarray):
        assert len(bias.shape) == 1
        length = len(bias)
        assert length == len(self.cells)
        for i in range(length):
            self.cells[i].bias = bias[i]
    
    @property
    def target(self) -> np.ndarray:
        return np.array([c.target for c in self.cells])
    
    @target.setter
    def target(self, target: np.ndarray):
        assert len(target.shape) == 1
        length = len(target)
        assert length == len(self.cells)
        for i in range(length):
            self.cells[i].target = target[i]
    
    def forward(self) -> np.ndarray:
        return np.array([c.forward() for c in self.cells])
    
    def backward(self) -> np.ndarray:
        return np.array([c.backward() for c in self.cells])
    
    def commit(self, rate):
        for c in self.cells:
            c.commit(rate)


class InputLayer(Layer):
    def __init__(self, length: int, name='', activate_function=Function.sigmoid()):
        super(InputLayer, self).__init__(length, name, activate_function)
        self.__data = np.zeros((length,))
    
    @property
    def data(self) -> np.ndarray:
        return self.__data
    
    @data.setter
    def data(self, data: np.ndarray):
        assert len(data.shape) == 1
        length = len(data)
        assert length == len(self.cells)
        self.__data = data
        for i in range(len(data)):
            self.cells[i].activate_function = Function.bias(self.__data[i])
