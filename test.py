# -*- coding: utf-8 -*-
from nn import Neuron, Function, Link, connect
import numpy as np

n1 = Neuron('n1', activate_function=Function.bias(0))
n2 = Neuron('n2', activate_function=Function.bias(0))
n3 = Neuron('n3')
n4 = Neuron('n4')
n5 = Neuron('n5')

ns = (n1, n2, n3, n4, n5)

connect(n1, n3)
connect(n1, n4)
connect(n2, n3)
connect(n2, n4)
connect(n3, n5)
connect(n4, n5)


def see(n: Neuron):
    for l in n.output_list:
        print(l)
        see(l.destination)


n1.activate_function = Function.bias(0)
n2.activate_function = Function.bias(0)
print(n5.forward())
n1.activate_function = Function.bias(0)
n2.activate_function = Function.bias(1)
print(n5.forward())
n1.activate_function = Function.bias(1)
n2.activate_function = Function.bias(0)
print(n5.forward())
n1.activate_function = Function.bias(1)
n2.activate_function = Function.bias(1)
print(n5.forward())

for k in range(10000):
    a, b = np.random.randint(2), np.random.randint(2)
    c = int(a == b)
    n1.activate_function = Function.bias(a)
    n2.activate_function = Function.bias(b)
    n5.target = c
    print(n5.forward(), n5.target, (a, b))
    # see(n1)
    for n in ns:
        n.commit(1.0)

n1.activate_function = Function.bias(0)
n2.activate_function = Function.bias(0)
print(n5.forward())
n1.activate_function = Function.bias(0)
n2.activate_function = Function.bias(1)
print(n5.forward())
n1.activate_function = Function.bias(1)
n2.activate_function = Function.bias(0)
print(n5.forward())
n1.activate_function = Function.bias(1)
n2.activate_function = Function.bias(1)
print(n5.forward())

see(n1)
see(n2)
