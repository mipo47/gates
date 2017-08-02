import numpy as np

class Optimizer:
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate

    def update(self, w, dW):
        w += dW * self.learning_rate

class Optimizer2:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.dY = 0

    def update(self, w, dW):
        batch_size = dW.shape[0]
        w += dW.T.dot(self.dY) * self.learning_rate / batch_size

class SGD:
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate

    def update(self, w, gW):
        w -= gW * self.learning_rate

class Momentum:
    def __init__(self, learning_rate = 1.0, momentum = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()

    def update(self, w, gW):
        w_id = id(w)
        if w_id in self.velocities:
            velocity = self.velocities[w_id]
        else:
            velocity = gW * self.learning_rate # by default used 0

        self.velocities[w_id] = velocity = velocity * self.momentum + gW * self.learning_rate
        w -=  velocity

class L1:
    def __init__(self, optimizer, regularization):
        self.optimizer = optimizer
        self.regularization = regularization

    def update(self, w, gW):
        self.optimizer.update(w, gW)
        w -= w * self.regularization