import numpy as np
from gates.gate import *


def generate_data(count=100, input_count=2, output_count=2, bias=True):
    x = (np.random.rand(count, input_count) - 0.5) * 2.0 * 10.0 # -10 to 10
    x = x.astype(dtype=Gate.TYPE)
    y = np.zeros((count, output_count), dtype=Gate.TYPE)

    y[:, 0] = -1 * x[:, 0] + 2 * x[:, 1] + (3 if bias else 0)
    if output_count >= 2:
        y[:, 1] = 1 * x[:, 0] - 4 * x[:, 1] + (-2 if bias else 0)
    if output_count >= 3:
        y[:, 2] = 4 * x[:, 0] - 2 * x[:, 1]
    if output_count >= 4:
        raise "too many output count, not supported"
    return x, y


def print_weights(layer, recursive = True):
    all_w = []

    while layer:
        if isinstance(layer, GateWeights):
            print("w = ", layer.w)
            all_w = np.concatenate((all_w, layer.w.reshape(-1)))
        layer = layer.prev
        if not recursive: break

    print('L2 reg w = ', np.mean(all_w ** 2)) #, np.min(all_w), np.max(all_w))


def checkGradient(gateWeights, loss, X, optimizer):
    temp_w = np.copy(gateWeights.w)
    temp_learning_rate = optimizer.learning_rate
    optimizer.learning_rate = 0.0

    oldLoss = loss.forward(X)
    dValue = np.ones_like(oldLoss)
    loss.backward(dValue, optimizer)

    d = 0.001
    i = np.random.randint(temp_w.shape[0])
    if temp_w.ndim == 1:
        grad = gateWeights.gW[i]
        gateWeights.w[i] += d
    elif temp_w.ndim == 2:
        j = np.random.randint(temp_w.shape[1])
        grad = gateWeights.gW[i, j]
        gateWeights.w[i, j] += d
    elif temp_w.ndim == 4: # convolutional filters
        i1 = np.random.randint(temp_w.shape[1])
        i2 = np.random.randint(temp_w.shape[2])
        i3 = np.random.randint(temp_w.shape[3])
        grad = gateWeights.gW[i, i1, i2, i3]
        gateWeights.w[i, i1, i2, i3] += d
    else:
        raise "dimensions are not supported"

    newLoss = loss.forward(X)

    realGrad = (newLoss - oldLoss) / d
    # realGrad = np.sum(realGrad, axis=0)

    diff = np.abs(realGrad - grad) / np.max(np.abs([realGrad, grad, 1e-7]))
    status = "OK" if diff < 0.05 else "Bad " + str(diff) + "\n  "
    print(status, "gradient real/expected", realGrad, grad)

    # print('g weights', gateWeights.gW)
    # print("old loss", oldLoss)
    # print("new loss", newLoss)

    # restore
    gateWeights.w = temp_w
    optimizer.learning_rate = temp_learning_rate

