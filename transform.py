import numpy as np
from gates.gate import *


class Reshape(Gate):
    def __init__(self, prev, output_shape):
        super().__init__(prev)
        self.input_shape = None  # will be known in forward pass
        self.output_shape = output_shape  # output shape excluding batch

    def forward(self, value):
        prev_value = self.prev.forward(value)

        # save for backward pass, exclude batch size
        self.input_shape = prev_value.shape[1:]

        self.value = prev_value.reshape((-1,) + self.output_shape)
        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = gValue.reshape((-1,) + self.input_shape)
        self.prev.backward(prev_gValue, optimizer)


class Flatten(Gate):
    def __init__(self, prev):
        super().__init__(prev)
        self.input_shape = None  # will be known in forward pass

    def forward(self, value):
        prev_value = self.prev.forward(value)

        # save for backward pass, exclude batch size
        self.input_shape = prev_value.shape[1:]

        self.value = prev_value.reshape((prev_value.shape[0], -1))
        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = gValue.reshape((-1,) + self.input_shape)
        self.prev.backward(prev_gValue, optimizer)

