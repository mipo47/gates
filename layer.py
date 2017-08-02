from gates.dot import *
from gates.bias import *
from gates.activations import *


class Layer(Gate):
    def __init__(self, prev, size, activation=Sigmoid):
        super().__init__(prev, size)
        net = prev
        self.dot = net = Dot2(net, size)
        self.bias = net = Bias(net)
        if activation is not None:
            self.activate = net = activation(net)
        self.net = net

    def forward(self, value):
        self.value = self.net.forward(value)
        return self.value

    def backward(self, dValue, optimizer):
        return self.net.backward(dValue, optimizer)