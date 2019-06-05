# -*- coding: utf-8 -*-
from nn import Layer, InputLayer, dense, Function, Initializer, Sequential
import numpy as np
import logging

logging.basicConfig(level=logging.NOTSET)


def see(layer: Layer):
    for n in layer.cells:
        print(n)
        for link in n.output_list:
            print(link)


for it in range(10):
    s = Sequential()
    s.add_layer(InputLayer(2, 'input'))
    s.add_layer(Layer(2, 'hidden', Function.ReLU(), Initializer.uniform(0, 0.1)), Initializer.uniform(-0.1, 0.1))
    s.add_layer(Layer(1, 'output', initializer=Initializer.uniform(0, 0.1)), Initializer.uniform(-0.1, 0.1))
    
    logging.info('TEST %i' % it)
    
    error_val = [1]
    error_val_max_length = 50
    while np.mean(error_val) > 0.05:
        a, b = np.random.randint(2), np.random.randint(2)
        c = int(a == b)
        input_data = np.array([a, b])
        target_output = np.array([c])
        s.commit(input_data, target_output, 0.1)
        error = s.forward(input_data)[0] - target_output
        error_val.append(np.sqrt(np.dot(error, error)))
        if len(error_val) > error_val_max_length:
            error_val.pop(0)
        print('error=%.5f' % np.mean(error_val), end='\r')
    
    logging.debug('-----------------s([0,0])=%s' % s.forward(np.array([0, 0]))[0])
    logging.debug('-----------------s([0,1])=%s' % s.forward(np.array([0, 1]))[0])
    logging.debug('-----------------s([1,0])=%s' % s.forward(np.array([1, 0]))[0])
    logging.debug('-----------------s([1,1])=%s' % s.forward(np.array([1, 1]))[0])
