import numpy as np
from gates.gate import Gate


def init_zero(shape):
    return np.zeros(shape, dtype=Gate.TYPE)


def xavier1(shape):
    range = np.sqrt(6 / (shape[0] + shape[1]))
    return np.random.uniform(-range, range, shape).astype(Gate.TYPE)


def xavier2(shape):
    return ((2 * np.random.random(shape) - 1) / np.sqrt(shape[0])).astype(Gate.TYPE)


# from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def xavier3(shape):
    return (np.random.randn(shape[0], shape[1]) / np.sqrt(shape[0])).astype(Gate.TYPE)


DEFAULT_DOT_INITIALIZATION = xavier3
DEFAULT_BIAS_INITIALIZATION = init_zero