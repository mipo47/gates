import numpy as np

class Gate:
    TYPE = np.float32

    def __init__(self, prev = None, size = None, value = None):
        self.prev = prev
        self.value = value
        if size is not None:
            self.size = size
        elif value is not None:
            self.size = value.shape[1]
        elif prev is not None:
            self.size = prev.size
        else:
            self.size = None

    def forward(self, value):
        if self.prev is not None:
            return self.prev.forward(value)

        self.value = value
        return value

    def backward(self, gValue, optimizer):
        if self.prev is not None:
            self.prev.backward(gValue, optimizer)


class GateWeights:
    def __init__(self):
        self.w = None
        self.gW = None


class DataGate(Gate):
    def __init__(self, value):
        super().__init__(value=value)

    def forward(self, value = None):
        return self.value

    def backward(self, gValue, optimizer):
        optimizer.update(self.value, gValue)


class GateW(Gate, GateWeights):
    def forward(self, value):
        self.w = self.value = value
        return value

    def backward(self, gValue, optimizer):
        self.gW = gValue