import numpy as np
from gates.gate import *


class Conv(Gate, GateWeights):
    def __init__(self, prev, in_shape, out_channels=10, filter_size=(3, 3), learning_rate=0.001):
        super().__init__(prev)
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.learning_rate = learning_rate

        padding = 0
        stride = 1

        out_h = int((in_shape[0] - filter_size[0] + 2 * padding) / stride + 1)
        out_w = int((in_shape[1] - filter_size[1] + 2 * padding) / stride + 1)
        self.output_shape = (out_h, out_w, out_channels)

        self.size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        filter_height, filter_width = filter_size
        in_channels = in_shape[2]

        # filters
        self.filter_shape = (filter_height, filter_width, in_channels, out_channels)

        self.w = np.random.randn(*self.filter_shape) / np.sqrt(out_channels * 0.5)

        # self.w = np.round((np.random.random(self.filter_shape) - 0.5) * 4)

        # self.w = np.zeros(self.filter_shape)
        # self.w[1, 1, 0, 0] = 1
        # self.w[0, 0, 1, 1] = 1
        # self.w[0, 0, 2, 1] = -1

        # print(np.transpose(self.w, (3,2,0,1)))
        # print("---------------------------")

    def forward(self, value):
        bchw = self.prev.forward(value)
        if bchw.shape[1:] != self.in_shape:
            raise "Invalid input format, needed: (batch, channel , height, width)"

        batch_size = bchw.shape[0]
        self.value = np.zeros((batch_size,) + self.output_shape)

        for h in range(self.output_shape[0]):
            for w in range(self.output_shape[1]):
                in_view = bchw[
                          :,
                          h: h + self.filter_size[0],
                          w: w + self.filter_size[1],
                          :
                          ].reshape((batch_size,) + self.filter_shape[:-1] + (1,))

                matrix = in_view * self.w
                # sum by h, w, in_c
                out = np.sum(matrix, axis=(1, 2, 3))
                self.value[:, h, w, :] += out

        return self.value

    def backward(self, gValue, optimizer):
        batch_size = gValue.shape[0]
        prev_gValue = np.zeros((batch_size,) + self.in_shape)

        bchw = self.prev.value
        self.gW = np.zeros_like(self.w)

        for h in range(self.output_shape[0]):
            for w in range(self.output_shape[1]):
                # previous layer gradient
                pixel = gValue[:, h, w, :].reshape((batch_size, 1, 1, 1, -1))
                out = np.sum(pixel * self.w, axis=4)
                prev_gValue[
                :,
                h: h + self.filter_size[0],
                w: w + self.filter_size[1],
                :] += out

                # filters gradient
                in_view = bchw[
                          :,
                          h: h + self.filter_size[0],
                          w: w + self.filter_size[1],
                          :
                          ].reshape((batch_size,) + self.filter_shape[:-1] + (1,))

                self.gW += np.sum(in_view * pixel, axis=0)

        optimizer.update(self.w, self.gW * self.learning_rate)
        self.prev.backward(prev_gValue, optimizer)
