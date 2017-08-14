import numpy as np
from gates.gate import *
from gates.im2col import *


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


class Conv2D(Gate, GateWeights):
    def __init__(self, prev, in_shape,
                 out_channels=10,
                 filter_size=(3, 3),
                 learning_rate=0.001,
                 padding=0,
                 stride=1):
        super().__init__(prev)
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.learning_rate = learning_rate

        self.padding = padding
        self.stride = stride

        out_h = int((in_shape[1] - filter_size[0] + 2 * padding) / stride + 1)
        out_w = int((in_shape[2] - filter_size[1] + 2 * padding) / stride + 1)
        self.output_shape = (out_channels, out_h, out_w)

        self.size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        filter_height, filter_width = filter_size
        in_channels = in_shape[0]

        # filters
        self.filter_shape = (out_channels, in_channels, filter_height, filter_width)

        self.w = np.random.randn(*self.filter_shape) / np.sqrt(out_channels * 0.5)

        # self.w = np.round((np.random.random(self.filter_shape) - 0.5) * 4)

        # self.w = np.zeros(self.filter_shape)
        # self.w[0, 0, 1, 1] = 1
        # self.w[1, 1, 0, 0] = 1
        # self.w[1, 2, 0, 0] = -1

        for i_o, o in enumerate(self.w):
            for i_i, i in enumerate(o):
                for i_h, h in enumerate(i):
                    for i_w, w in enumerate(h):
                        self.w[i_o, i_i, i_h, i_w] = \
                            i_o + (i_i + 1) * 0.1 + (i_h + 1) * 0.01 + (i_w + 1) * 0.001

        # print(self.w)
        # print("---------------------------")

    def forward(self, value):
        # (10, 3, 4, 5) or (batch, in_c, in_h, in_w)

        input = self.prev.forward(value)
        if input.shape[1:] != self.in_shape:
            raise "Invalid input format, needed: (batch, channel , height, width)"

        # (2, 27) or (out_c, in_c * f_h * f_w)
        w_flat = self.w.reshape(self.out_channels, -1)

        # (27, 60) or (in_c * in_h * in_w, batch * out_h * out_w)
        self.input_col = im2col_indices(
            input, self.filter_size[0], self.filter_size[1],
            stride=self.stride, padding=self.padding)

        # (2, 60) or (output_channel, (batch x out_h, out_w))
        out = w_flat @ self.input_col

        # (2, 10, 2, 3) or (out_c, batch, out_h, out_w)
        out = out.reshape((self.out_channels, -1) + self.output_shape[1:])

        # (10, 2, 2, 3) or (batch, out_c, out_h, out_w)
        self.value = out.transpose(1, 0, 2, 3)
        return self.value

    def backward(self, gValue, optimizer):
        # gValue: (10, 2, 2, 3) or (batch, out_c, out_h, out_w)
        batch_size = gValue.shape[0]

        # (2, 60) or (out_c, batch * out_h * out_w)
        dout_flat = gValue.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

        # (2, 3, 3, 3) or (out_c, in_c, f_h, f_w)
        self.gW = (dout_flat @ self.input_col.T).reshape(self.w.shape)

        # (27, 2) or (in_c * f_h * f_w, out_c)
        w_flat_T = self.w.reshape(self.out_channels, -1).T

        # (27, 60) or (in_c * f_h * f_w, batch * out_h * out_w)
        dX_col = w_flat_T @ dout_flat

        # (10, 3, 4, 5) or (batch, in_c, in_h, in_w)
        shape = (batch_size,) + self.in_shape

        # (10, 3, 4, 5) or (batch, in_c, in_h, in_w)
        prev_gValue = col2im_indices(dX_col, shape, self.filter_size[0],
                                     self.filter_size[1], self.padding, self.stride)

        optimizer.update(self.w, self.gW * self.learning_rate)
        self.prev.backward(prev_gValue, optimizer)
