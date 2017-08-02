import numpy as np
from gates.gate import *


class Abs(Gate):
    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.abs(prev_value)
        return self.value

    def backward(self, dValue, optimizer):
        prev_value = self.prev.value
        dValue_prev = np.sign(prev_value) * dValue
        self.prev.backward(dValue_prev, optimizer)


class Plus(Gate):
    def __init__(self, prev, constant):
        super().__init__(prev)
        self.constant = constant

    def forward(self, value):
        prev_value = self.prev.forward(value)
        add = self.constant.forward(value) if isinstance(self.constant, Gate) else self.constant
        self.value = prev_value + add
        return self.value

    def backward(self, dValue, optimizer):
        if isinstance(self.constant, Gate):
            self.constant.backward(dValue, optimizer)
        self.prev.backward(dValue, optimizer)


class Minus(Plus):
    def __init__(self, prev, constant):
        super().__init__(prev, -constant)


class Mult(Gate):
    def __init__(self, prev, constant):
        super().__init__(prev)
        self.constant = constant

    def forward(self, value):
        prev_value = self.prev.forward(value)
        mult = self.constant.forward(value) if isinstance(self.constant, Gate) else self.constant
        self.value = prev_value * mult
        return self.value

    def backward(self, gValue, optimizer):
        if isinstance(self.constant, Gate):
            self.constant.backward(gValue * self.prev.value, optimizer)
        self.prev.backward(gValue * self.constant, optimizer)


class Add(Gate, GateWeights):
    def __init__(self, prev):
        super().__init__(prev)
        self.w = np.array([0.0])

    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = prev_value + self.w
        return self.value

    def backward(self, gValue, optimizer):
        self.gW = np.array([np.sum(gValue)])
        optimizer.update(self.w, self.gW)
        self.prev.backward(gValue, optimizer)


class Mean(Gate):
    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.mean(prev_value)
        return self.value

    def backward(self, dValue = None, optimizer = None):
        # compute total overaged values in previous layer
        prev_value = self.prev.value
        prev_shape = prev_value.shape
        count = prev_shape[0] * prev_shape[1]

        dValue_prev = dValue / count
        self.prev.backward(dValue_prev, optimizer)


class Power(Gate):
    def __init__(self, prev, constant):
        super().__init__(prev)
        self.constant = constant

    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.power(prev_value, self.constant)
        return self.value

    def backward(self, dValue, optimizer):
        prev_value = self.prev.value

        # derivate of x^n = n * x^(n-1)
        if self.constant == 2: # simplify for power of 2
            dValue_prev = dValue * (2 * prev_value)
        else:
            dValue_prev = dValue * (self.constant * np.power(prev_value, self.constant - 1))

        self.prev.backward(dValue_prev, optimizer)