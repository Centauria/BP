# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
from nn.concept import Forward, Backward, Adaptable
from nn.function import Function, Initializer


class Neuron(Forward, Backward, Adaptable):
    def __init__(self, name='', bias: Optional[np.float] = None,
                 activate_function: Function = Function.sigmoid(),
                 initializer: () = Initializer.uniform()):
        # input_list_values: the links that connect to this neuron
        self.__input_list = []
        # output: the links that this neuron connects to
        self.__output_list = []
        self.__f: Function = activate_function
        self.name = name
        # short-circuit variables
        self.__bias: np.float = bias if bias is not None else initializer()
        # cache variables for temporary calculation, set to None after each commit
        self.__target: Optional[np.float] = None
        self.__delta_b: Optional[np.float] = None
    
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
    def v(self):
        return self.__bias + np.sum(self.input_list_values)
    
    @property
    def grad(self):
        return self.__f.d(self.__bias + np.sum(self.input_list_values))
    
    @property
    def delta_b(self):
        return self.__delta_b
    
    @property
    def target(self) -> np.float:
        return self.__target
    
    @target.setter
    def target(self, target: Optional[np.float]):
        self.__target = target
        self.build_cache()
    
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
        if self.__target is None:
            delta_result = np.sum([l.backward() for l in self.__output_list])
        else:
            delta_result = self.__target - self.forward()
        return delta_result
    
    def commit(self, rate):
        for link in self.__input_list:
            link.commit(rate)
        self.__bias += rate * self.__delta_b
        self.clear_cache()
    
    def build_cache(self):
        self.__delta_b = self.backward() * self.grad
        for link in self.__input_list:
            link.build_cache()
    
    def clear_cache(self):
        self.__target = None
        self.__delta_b = None
    
    def connect(self, other, weight=None):
        from nn.connection import connect
        return connect(self, other, weight)
    
    def disconnect(self, other=None):
        from nn.connection import disconnect
        return disconnect(self, other)


class Link(Forward, Backward, Adaptable):
    def __init__(self, source: Optional[Neuron], destination: Optional[Neuron], weight=0.0):
        self.source = source
        self.destination = destination
        self.weight = weight
        # cache variables for temporary calculation, set to None after each commit
        self.__delta_w: Optional[np.float] = None
    
    def __repr__(self):
        return '%s <--%.10f--> %s' % (self.source, self.weight, self.destination)
    
    def forward(self):
        return self.source.forward() * self.weight
    
    def backward(self):
        return self.destination.backward() * self.weight

    def commit(self, rate):
        self.weight += rate * self.__delta_w
        self.clear_cache()

    def build_cache(self):
        self.__delta_w = self.source.forward() * self.destination.delta_b

    def clear_cache(self):
        self.__delta_w = None
