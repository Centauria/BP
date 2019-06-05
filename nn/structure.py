# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, List
from nn.concept import Forward, Backward, Adaptable
from nn.element import Neuron
from nn.function import Function, Initializer


class Layer(Forward, Backward, Adaptable):
    def __init__(self, length: int, name='',
                 activate_function=Function.sigmoid(),
                 initializer: () = Initializer.uniform()):
        self.name = name
        self.cells: List[Neuron] = [
            Neuron('_'.join((self.name, str(i))),
                   activate_function=activate_function,
                   initializer=initializer)
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
    
    def build_cache(self):
        for c in self.cells:
            c.build_cache()
    
    def clear_cache(self):
        for c in self.cells:
            c.clear_cache()


class InputLayer(Layer):
    def __init__(self, length: int, name=''):
        super(InputLayer, self).__init__(length, name, Function.bias(0))
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


class Model(Forward):
    def __init__(self):
        self.layers: List[Layer] = []
        self.input_layers: List[InputLayer] = []
        self.output_layers: List[Layer] = []
    
    def forward(self, *data):
        assert len(data) == len(self.input_layers)
        for i in range(len(data)):
            self.input_layers[i].data = data[i]
        return [self.output_layers[i].forward() for i in range(len(self.output_layers))]


class Sequential(Model, Adaptable):
    def __init__(self):
        super(Sequential, self).__init__()
        self.input_layer: Optional[InputLayer] = None
        self.output_layer: Optional[Layer] = None
    
    def add_layer(self, layer: Layer, input_weight_initializer: () = Initializer.uniform(-1, 1)):
        from nn.connection import dense
        if len(self.layers) > 0:
            dense(self.layers[-1], layer, input_weight_initializer)
        else:
            if not isinstance(layer, InputLayer):
                raise ValueError
        self.layers.append(layer)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.input_layers = self.layers[:1]
        self.output_layers = self.layers[-1:]
    
    def commit(self, input_data, target_output, rate):
        self.forward(input_data)
        self.target = target_output
        for layer in self.layers[1:]:
            layer.build_cache()
        for layer in self.layers[1:]:
            layer.commit(rate)
    
    @property
    def target(self) -> np.ndarray:
        return np.array([c.target for c in self.output_layer.cells])
    
    @target.setter
    def target(self, target: np.ndarray):
        assert len(target.shape) == 1
        length = len(target)
        assert length == len(self.output_layer.cells)
        for i in range(length):
            self.output_layer.cells[i].target = target[i]
