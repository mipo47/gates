import numpy as np
from gates.gate import Gate

class Sigmoid(Gate):
    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = 1 / (1 + np.exp(-prev_value))
        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = self.value * (1 - self.value) * gValue
        self.prev.backward(prev_gValue, optimizer)


class Tanh(Gate):
    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.tanh(prev_value)
        return self.value

    def backward(self, gValue, optimizer):
        prev_value = self.prev.value
        prev_gValue = gValue * (1.0 - np.tanh(prev_value)**2)
        self.prev.backward(prev_gValue, optimizer)


class Relu(Gate):
    def forward(self, value):
        prev_value = self.prev.forward(value)
        self.value = np.maximum(0, prev_value)
        return self.value

    def backward(self, gValue, optimizer):
        prev_value = self.prev.value
        prev_gValue = (prev_value > 0) * gValue
        self.prev.backward(prev_gValue, optimizer)


# original: https://github.com/martinkersner/cs231n/blob/master/assignment1/softmax.py
class Softmax(Gate):
    def forward(self, value):
        num_train = value.shape[0]

        prev_value = self.prev.forward(value)
        exps = np.exp(prev_value - np.max(prev_value, axis=1, keepdims=True))
        sum = np.sum(exps, axis=1, keepdims=True)
        prob_scores = exps / sum

        # normalize
        self.value = prob_scores / np.sum(prob_scores, axis=1, keepdims=True)
        return self.value

    def backward(self, gValue, optimizer):
        raise "Not implemented"

