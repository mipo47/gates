import numpy as np
from gates.gate import *


class Bias(Gate, GateWeights):
    def __init__(self, prev):
        super().__init__(prev)
        self.w = np.zeros(prev.size)

    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = prev_value + self.w
        return self.value

    def backward(self, gValue, optimizer):
        self.gW = np.sum(gValue, axis=0)
        optimizer.update(self.w, self.gW)
        self.prev.backward(gValue, optimizer)