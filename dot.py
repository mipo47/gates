import numpy as np
from gates.gate import *


class Dot(Gate):
    def __init__(self, prev, w):
        super().__init__(prev)

        if isinstance(w, int):
            w = (prev.size, w)

        if isinstance(w, tuple):
            w = (2 * np.random.random(w) - 1) / np.sqrt(w[0])

        self.w = w
        self.size = w.shape[1]

    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.dot(prev_value, self.w)
        return self.value

    def backward(self, dValue, optimizer):
        prev_value = self.prev.value

        dW = prev_value.T.dot(dValue) / prev_value.shape[0]
        optimizer.update(self.w, dW)

        prevGrad = dValue.dot(self.w.T)
        self.prev.backward(prevGrad, optimizer)


class Dot2(Gate, GateWeights):
    def __init__(self, prev, w):
        super().__init__(prev)

        if isinstance(w, int):
            w = (prev.size, w)

        if isinstance(w, tuple):
            w = (2 * np.random.random(w) - 1) / np.sqrt(w[0])

        self.w = w.astype(Gate.TYPE)
        self.size = w.shape[1]

    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.dot(prev_value, self.w)
        return self.value

    def backward(self, gValue, optimizer):
        prev_value = self.prev.value

        prev_dValue = gValue.dot(self.w.T)

        self.gW = prev_value.T.dot(gValue)
        optimizer.update(self.w, self.gW)

        self.prev.backward(prev_dValue, optimizer)