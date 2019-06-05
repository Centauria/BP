# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, Optional
from nn.structure import Layer
from nn.element import Neuron, Link
from nn.function import Initializer


def connect(neuron_1: Neuron, neuron_2: Neuron,
            weight: Optional[np.float] = None,
            initializer: () = Initializer.uniform(-1, 1)) -> Link:
    weight = weight if weight is not None else initializer()
    link = Link(neuron_1, neuron_2, weight)
    neuron_1.output_list.append(link)
    neuron_2.input_list.append(link)
    return link


def disconnect(neuron_or_layer_1: Union[Neuron, Layer], neuron_or_layer_2: Union[Neuron, Layer, None] = None):
    if neuron_or_layer_2 is None:
        if isinstance(neuron_or_layer_1, Neuron):
            neuron_or_layer_1.input_list.clear()
            neuron_or_layer_1.output_list.clear()
        elif isinstance(neuron_or_layer_1, Layer):
            for n in neuron_or_layer_1.cells:
                disconnect(n)
    else:
        if isinstance(neuron_or_layer_1, Neuron) and isinstance(neuron_or_layer_2, Neuron):
            for link in neuron_or_layer_1.output_list:
                if link.destination == neuron_or_layer_2:
                    neuron_or_layer_1.output_list.remove(link)
            for link in neuron_or_layer_2.input_list:
                if link.source == neuron_or_layer_1:
                    neuron_or_layer_2.input_list.remove(link)
        elif isinstance(neuron_or_layer_1, Layer):
            for n in neuron_or_layer_1.cells:
                disconnect(n, neuron_or_layer_2)
        elif isinstance(neuron_or_layer_2, Layer):
            for n in neuron_or_layer_2.cells:
                disconnect(neuron_or_layer_1, n)


def dense(layer_1: Layer, layer_2: Layer, initializer=Initializer.uniform(-1, 1)):
    for c_1 in layer_1.cells:
        for c_2 in layer_2.cells:
            connect(c_1, c_2, initializer=initializer)
