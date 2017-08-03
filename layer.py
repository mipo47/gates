from gates.dot import *
from gates.bias import *
from gates.activations import *


def Layer(prev, size, activation=Sigmoid):
    net = Dot2(prev, size)
    net = Bias(net)

    if activation is not None:
        net = activation(net)

    return net
