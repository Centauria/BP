# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
from nn.structure import Layer
from nn.element import Neuron, Link


def connect(neuron_1: Neuron, neuron_2: Neuron, weight: Optional[np.float] = None) -> Link:
    weight = weight if weight is not None else np.random.rand()
    link = Link(neuron_1, neuron_2, weight)
    neuron_1.output_list.append(link)
    neuron_2.input_list.append(link)
    return link


def dense(layer_1: Layer, layer_2: Layer):
    for c_1 in layer_1.cells:
        for c_2 in layer_2.cells:
            connect(c_1, c_2)
