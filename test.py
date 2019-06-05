# -*- coding: utf-8 -*-
from nn import Layer, InputLayer, dense, Function
import numpy as np
import logging

logging.basicConfig(level=logging.NOTSET)


def see(layer: Layer):
    for n in layer.cells:
        print(n)
        for link in n.output_list:
            print(link)


for it in range(10):
    l1 = InputLayer(2, 'input')
    l2 = Layer(2, 'hidden1')
    l3 = Layer(1, 'output')
    
    ls = [l2, l3]
    
    dense(l1, l2)
    dense(l2, l3)
    
    for k in range(20000):
        a, b = np.random.randint(2), np.random.randint(2)
        c = int(a == b)
        l1.data = np.array([a, b])
        l3.target = np.array([c])
        # logging.debug(l3.target - l3.forward())
        # see(n1)
        for l in ls:
            l.build_cache()
        for l in ls:
            l.commit(0.5)
    
    logging.info('TEST %i' % it)
    l1.data = np.array([0, 0])
    logging.debug(l3.forward())
    l1.data = np.array([0, 1])
    logging.debug(l3.forward())
    l1.data = np.array([1, 0])
    logging.debug(l3.forward())
    l1.data = np.array([1, 1])
    logging.debug(l3.forward())

see(l1)
see(l2)
see(l3)
