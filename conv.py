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
        self.filter_count = out_channels
        self.filter_size = filter_size

        self.in_shape = in_shape
        self.output_shape = (out_channels,
                             in_shape[1] - self.filter_size[0] + 1,
                             in_shape[2] - self.filter_size[1] + 1)

        self.size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        filter_height, filter_width = filter_size
        in_channels = in_shape[0]
        self.filters = np.zeros([out_channels, in_channels, filter_height, filter_width])

        self.filters[0, 0, 1, 1] = 1
        self.filters[1, 1, 0, 0] = 1
        self.filters[1, 2, 0, 0] = -1

    def forward(self, value):
        bchw = self.prev.forward(value)
        if bchw.shape[1:] != self.in_shape:
            raise "Invalid input format, needed: (batch, channel , height, width)"

        batch_size = bchw.shape[0]
        self.value = np.zeros((batch_size,) + self.output_shape)
        # self.value = bhwc[:, 1:-1, 1:-1, :]

        for h in range(self.output_shape[1]):
            for w in range(self.output_shape[2]):
                in_view = bchw[
                          :, :,
                          h: h + self.filter_size[0],
                          w: w + self.filter_size[1]]

                for out_c in range(self.output_shape[0]):
                    # calculate filter applied matrix
                    matrix = in_view * self.filters[out_c]
                    out = np.sum(matrix, axis=(1, 2, 3))
                    # if np.sum(out) > 0:
                    #     print(h, w, in_c, out_c)
                    #     print(bhwc[
                    #          0,
                    #          h:h + self.filter_size[0],
                    #          w:w + self.filter_size[1],
                    #          in_c])

                    self.value[:, out_c, h, w] += out

        return self.value

    def backward(self, gValue, optimizer):
        prev_gValue = np.zeros((gValue.shape[0],) + self.in_shape)
        self.prev.backward(prev_gValue, optimizer)
