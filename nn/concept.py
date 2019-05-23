# -*- coding: utf-8 -*-
import abc


class Forward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass


class Backward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self):
        pass


class Adaptable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def commit(self, rate):
        pass
