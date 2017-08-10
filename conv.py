import numpy as np
from gates.gate import *


class Reshape(Gate):
    def __init__(self, prev, output_shape):
        super().__init__(prev)
        self.input_shape = None  # will be known in forward pass
        self.output_shape = output_shape  # output shape excluding batch

    def forward(self, value):
        prev_value = self.prev.forward(value)

        # save for backward pass, exclude batch size
        self.input_shape = prev_value.shape[1:]

        self.value = prev_value.reshape((-1,) + self.output_shape)
        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = gValue.reshape((-1,) + self.input_shape)
        self.prev.backward(prev_gValue, optimizer)


class Flatten(Gate):
    def __init__(self, prev):
        super().__init__(prev)
        self.input_shape = None  # will be known in forward pass

    def forward(self, value):
        prev_value = self.prev.forward(value)

        # save for backward pass, exclude batch size
        self.input_shape = prev_value.shape[1:]

        self.value = prev_value.reshape((prev_value.shape[0], -1))
        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = gValue.reshape((-1,) + self.input_shape)
        self.prev.backward(prev_gValue, optimizer)


class Conv(Gate):
    def __init__(self, prev, in_shape, out_channels=10, filter_size=(3, 3)):
        super().__init__(prev)
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.filter_size = filter_size

        padding = 0
        stride = 1

        out_h = int((in_shape[1] - filter_size[0] + 2 * padding) / stride + 1)
        out_w = int((in_shape[2] - filter_size[1] + 2 * padding) / stride + 1)
        self.output_shape = (out_channels, out_h, out_w)

        self.size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        filter_height, filter_width = filter_size
        in_channels = in_shape[0]

        # self.filters = np.zeros([out_channels, in_channels, filter_height, filter_width])
        # self.filters[0, 0, 1, 1] = 1
        # self.filters[1, 1, 0, 0] = 1
        # self.filters[1, 2, 0, 0] = -1

        self.filters = np.round((np.random.random([out_channels, in_channels, filter_height, filter_width]) - 0.5) * 4)

        # print(self.filters)
        # print("---------------------------")

    def forward(self, value):
        bchw = self.prev.forward(value)
        if bchw.shape[1:] != self.in_shape:
            raise "Invalid input format, needed: (batch, channel , height, width)"

        batch_size = bchw.shape[0]
        self.value = np.zeros((batch_size,) + self.output_shape)
        # self.value = bhwc[:, 1:-1, 1:-1, :]

        for b in range(batch_size):
            for h in range(self.output_shape[1]):
                for w in range(self.output_shape[2]):
                    in_view = bchw[
                              b, :,
                              h: h + self.filter_size[0],
                              w: w + self.filter_size[1]]
                    out = np.sum(in_view * self.filters, axis=(1, 2, 3))
                    self.value[b, :, h, w] += out

        return self.value

    def backward(self, gValue, optimizer):
        batch_size = gValue.shape[0]
        prev_gValue = np.zeros((batch_size,) + self.in_shape)

        for b in range(batch_size):
            for h in range(self.output_shape[1]):
                for w in range(self.output_shape[2]):
                    pixel = gValue[b, :, h, w].reshape((-1, 1, 1, 1))
                    out = np.sum(pixel * self.filters, axis=0)
                    prev_gValue[
                        b, :,
                        h: h + self.filter_size[0],
                        w: w + self.filter_size[1]] += out

        self.prev.backward(prev_gValue, optimizer)
