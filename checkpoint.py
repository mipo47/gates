import numpy as np
from gates.gate import *


class Checkpoint:
    def __init__(self, net):
        self.net = net
        self.weights = None

    def backup(self):
        self.weights = []
        gate = self.net
        while gate is not None:
            if isinstance(gate, GateWeights):
                self.weights.append(np.copy(gate.w))
            gate = gate.prev

    def restore(self):
        gate = self.net
        i = 0
        while gate is not None:
            if isinstance(gate, GateWeights):
                gate.w = self.weights[i]
                i += 1
            gate = gate.prev

    def copy_to(self, dest):
        gate = self.net
        while gate is not None:
            if isinstance(gate, GateWeights):
               dest.w = np.copy(gate.w)
            gate = gate.prev
            dest = dest.prev