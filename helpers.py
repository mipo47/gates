import numpy as np
from gates.gate import *


def generate_data(count=100, input_count=2, output_count=2, bias=True):
    x = np.random.rand(count, input_count)
    y = np.zeros((count, output_count))
    y[:, 0] = -1 * x[:, 0] + 2 * x[:, 1] + (3 if bias else 0)
    if output_count >= 2:
        y[:, 1] = x[:, 0] - 4 * x[:, 1] + (-2 if bias else 0)
    if output_count >= 3:
        y[:, 2] = 4 * x[:, 0] - 2 * x[:, 1]
    if output_count >= 4:
        raise "too many output count, not supported"
    return x, y


def print_weights(layer):
    while layer:
        if isinstance(layer, GateWeights):
            print("w = ", layer.w)
        layer = layer.prev


def checkGradient(gateWeights, loss, X, optimizer):
    temp_w = np.copy(gateWeights.w)
    temp_learning_rate = optimizer.learning_rate
    optimizer.learning_rate = 0.0

    oldLoss = loss.forward(X)
    dValue = np.ones_like(oldLoss)
    loss.backward(dValue, optimizer)

    d = 0.00001
    i = np.random.randint(temp_w.shape[0])
    j = None
    if temp_w.ndim == 1:
        grad = gateWeights.gW[i]
    else:  # 2 dimensions
        j = np.random.randint(temp_w.shape[1])
        grad = gateWeights.gW[i, j]

    if temp_w.ndim == 1:
        gateWeights.w[i] += d
    else: # 2 dimensions
        gateWeights.w[i, j] += d
    newLoss = loss.forward(X)

    realGrad = (newLoss - oldLoss) / d
    # realGrad = np.sum(realGrad, axis=0)

    diff = np.abs(realGrad - grad) / np.max(np.abs([realGrad, grad, 1e-7]))
    status = "OK" if diff < 0.0025 else "Bad " + str(diff) + "\n  "
    print(status, "gradient real/expected", realGrad, grad)

    # print('g weights', gateWeights.gW)
    # print("old loss", oldLoss)
    # print("new loss", newLoss)

    # restore
    gateWeights.w = temp_w
    optimizer.learning_rate = temp_learning_rate

