# -*- coding: utf-8 -*-
import nn
import numpy as np

n1 = nn.Neuron('A')
n2 = nn.Neuron('B')
link = nn.connect(n1, n2, 0.5)

print('n1: %s, n2: %s, link: %s' % (n1, n2, link))
print(n1.forward())
print(link.forward())
print(n2.forward())
print('BP:')
print(n2.backward(1))
print(n1.backward())
