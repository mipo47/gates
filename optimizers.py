import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1.0):
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
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update(self, w, gW):
        w -= gW * self.learning_rate


class Momentum:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()

    def update(self, w, gW):
        w_id = id(w)
        if w_id in self.velocities:
            velocity = self.velocities[w_id]
        else:
            velocity = gW * self.learning_rate  # by default used 0

        self.velocities[w_id] = velocity = velocity * self.momentum + gW * self.learning_rate
        w -= velocity


# https://medium.com/@nishantnikhil/adam-optimizer-notes-ddac4fd7218
class Adam:
    def __init__(self, learning_rate=0.001, b1=0.001, b2=0.9, e=1e-8):
        self.learning_rate = learning_rate
        self.b1, self.b2, self.e = b1, b2, e

        self.reset()

    def update(self, w, gW):
        w_id = id(w)
        m1 = self.m1[w_id] if w_id in self.m1 else gW
        m2 = self.m2[w_id] if w_id in self.m2 else gW**2

        self.m1[w_id] = m1 = self.b1 * m1 + (1.0 - self.b1) * gW
        self.m2[w_id] = m2 = self.b2 * m2 + (1.0 - self.b2) * (gW**2)

        # bias correction
        m1 = m1 / (1.0 - self.b1)
        m2 = m2 / (1.0 - self.b2)

        w -= self.learning_rate * m1 / (np.sqrt(m2) + self.e)

    def reset(self):
        self.m1 = dict()
        self.m2 = dict()