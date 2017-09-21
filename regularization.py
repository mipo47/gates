import numpy as np


class Regularizer:
    def __init__(self, optimizer, regularization):
        self.optimizer = optimizer
        self.regularization = regularization

    def update(self, w, gW):
        gW += w * self.regularization
        self.optimizer.update(w, gW)


class RegularizerDir:
    def __init__(self, optimizer, regularization):
        self.optimizer = optimizer
        self.regularization = regularization

    def update(self, w, gW):
        mult = 1.0 + np.abs(w) * self.regularization
        gW *= mult ** (np.sign(gW) * np.sign(w))
        self.optimizer.update(w, gW)

# class RegularizerL2:
#     def __init__(self, optimizer, regularization):
#         self.optimizer = optimizer
#         self.regularization = regularization
#
#     def update(self, w, gW):
#         self.optimizer.update(w, gW)
#         w -= (gW ** 2) * self.regularization