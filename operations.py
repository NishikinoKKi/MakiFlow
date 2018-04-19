#!/usr/bin/env python
# -*- coding: utf-8 -*-
from queue import Queue

import numpy as np


class Operation(object):
    def __init__(self, *input_nodes, name=None):
        self.input_nodes = input_nodes
        self.output_nodes = []

        self.output_value = None

        self.name = name

        self.graph = DEFAULT_GRAPH

        for node in input_nodes:
            node.output_nodes.append(self)

        self.graph.operations.append(self)

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self, grad=None):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


class Add(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


def add(x, y, name=None):
    return Add(x, y, name)


class Multiply(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad*y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad*x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


def multiply(x, y, name=None):
    return Multiply(x, y, name)


class MatMul(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        dfdx = np.dot(grad, np.transpose(y))
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]


def matmul(x, y, name=None):
    return MatMul(x, y, name)


class Sigmoid(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = 1/(1 + np.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad*self.output_value*(1 - self.output_value)


def sigmoid(x, name=None):
    return Sigmoid(x, name=name)


class Log(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.log(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad*1/x


def log(x, name=None):
    return Log(x, name=name)


class Negative(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -grad


class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


def reduce_sum(x, axis=None):
    return ReduceSum(x, axis=axis)


class Square(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        return grad*np.multiply(2.0, input_value)


def square(x, name=None):
    return Square(x, name=name)


class Constant(object):
    def __init__(self, value, name=None):
        self.value = value

        self.output_value = None

        self.output_nodes = []

        self.name = name

        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def constant(value, name=None):
    return Constant(value, name=name)


class Variable(object):
    def __init__(self, initial_value=None, name=None, trainable=True):
        self.initial_value = initial_value

        self.output_value = None

        self.output_nodes = []

        self.name = name

        self.graph = DEFAULT_GRAPH

        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


class Placeholder(object):
    def __init__(self, name=None):
        self.output_value = None

        self.output_nodes = []

        self.name = name

        self.graph = DEFAULT_GRAPH

        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def placeholder(name=None):
    return Placeholder(name=name)


def compute_gradients(target_op):
    grad_table = {}
    grad_table[target_op] = np.ones_like(target_op.output_value)
    queue = Queue()
    queue.put(target_op)

    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()
        if node != target_op:
            grads_wrt_node_output = []

            for output_node in node.output_nodes:
                grad_wrt_output_node_output = grad_table[output_node]

                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table

