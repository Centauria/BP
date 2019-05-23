# -*- coding: utf-8 -*-
from nn import Layer, InputLayer, dense
import numpy as np

l1 = InputLayer(2, 'input')
l2 = Layer(2, 'hidden')
l3 = Layer(1, 'output')

ls = (l1, l2, l3)

dense(l1, l2)
dense(l2, l3)


def see(layer: Layer):
    for n in layer.cells:
        print(n)
        for link in n.output_list:
            print(link)


l1.data = np.array([0, 0])
print(l3.forward())
l1.data = np.array([0, 1])
print(l3.forward())
l1.data = np.array([1, 0])
print(l3.forward())
l1.data = np.array([1, 1])
print(l3.forward())

for k in range(10000):
    a, b = np.random.randint(2), np.random.randint(2)
    c = int(a == b)
    l1.data = np.array([a, b])
    l3.target = np.array([c])
    print(l3.forward(), l3.target, (a, b))
    # see(n1)
    for l in ls:
        l.commit(1.0)

l1.data = np.array([0, 0])
print(l3.forward())
l1.data = np.array([0, 1])
print(l3.forward())
l1.data = np.array([1, 0])
print(l3.forward())
l1.data = np.array([1, 1])
print(l3.forward())

see(l1)
see(l2)
see(l3)
