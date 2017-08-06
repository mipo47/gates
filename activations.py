import numpy as np
from gates.gate import *


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
        prev_gValue = gValue * (1.0 - np.tanh(prev_value) ** 2)
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


# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
class BatchNorm(Gate, GateWeights):
    def __init__(self, prev):
        super().__init__(prev)

        self.w = np.array([1.0, 0.0], dtype=np.float32)
        self.eps = 1e-8

        self.cache = None

    def forward(self, value):
        x = self.prev.forward(value)

        # step1: calculate mean
        mu = np.mean(x, axis=0)

        # step2: subtract mean vector of every trainings example
        self.xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = self.xmu ** 2

        # step4: calculate variance
        self.var = np.mean(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        self.sqrtvar = np.sqrt(self.var + self.eps)

        # step6: invert sqrtwar
        self.ivar = 1. / self.sqrtvar

        # step7: execute normalization
        xhat = self.xmu * self.ivar

        # step8: Nor the two transformation steps
        gammax = self.w[0] * xhat

        # step9
        self.value = gammax + self.w[1]

        return self.value

    def backward(self, gValue, optimizer):
        N, D = gValue.shape

        # calculate instead of storing because it's too big
        xhat = self.xmu * self.ivar

        # step9
        dbeta = np.sum(gValue, axis=0)
        dgammax = gValue  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * self.w[0]

        # step7
        divar = np.sum(dxhat * self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar

        # step6
        dsqrtvar = -1. / (self.sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self.var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * self.xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        prev_gValue = dx1 + dx2

        # update gamma and beta
        self.gW = np.array([np.sum(dgamma), np.sum(dbeta)])
        optimizer.update(self.w, self.gW)

        self.prev.backward(prev_gValue, optimizer)
