# -*- coding: utf-8 -*-
import abc


class Forward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError


class Backward(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self):
        raise NotImplementedError


class Adaptable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def commit(self, rate):
        raise NotImplementedError
