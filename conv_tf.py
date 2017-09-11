import numpy as np
import tensorflow as tf
from gates.gate import *


class TF:
    sess = tf.Session(
        config=tf.ConfigProto(device_count = {'GPU': 0})
    )

    init = lambda: TF.sess.run(tf.global_variables_initializer())


class Conv_tf(Gate, GateWeights):
    def __init__(self, prev, in_shape,
                 out_channels=10,
                 filter_size=(3, 3),
                 learning_rate=0.001,
                 padding=0,
                 stride=1,
                 data_format="NHWC"): # "NCHW"
        super().__init__(prev)
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.learning_rate = learning_rate

        self.padding = padding
        self.stride = stride

        out_h = int((in_shape[0] - filter_size[0] + 2 * padding) / stride + 1)
        out_w = int((in_shape[1] - filter_size[1] + 2 * padding) / stride + 1)
        self.output_shape = (out_h, out_w, out_channels)

        self.size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        filter_height, filter_width = filter_size
        in_channels = in_shape[2]

        # filters
        self.filter_shape = (out_channels, in_channels, filter_height, filter_width)

        self.w = np.random.randn(*self.filter_shape) / np.sqrt(out_channels * 0.5)
        for i_o, o in enumerate(self.w):
            for i_i, i in enumerate(o):
                for i_h, h in enumerate(i):
                    for i_w, w in enumerate(h):
                        self.w[i_o, i_i, i_h, i_w] = \
                            i_o + (i_i + 1) * 0.1 + (i_h + 1) * 0.01 + (i_w + 1) * 0.001

        self.x = tf.placeholder(tf.float32, (None,) + in_shape, "inputs")

        # [filter_height, filter_width, in_channels, out_channels]
        self.filter = tf.Variable(self.w.transpose(2, 3, 1, 0), dtype=tf.float32)
        self.conv = tf.nn.conv2d(self.x, self.filter, [1, stride, stride, 1], "VALID", data_format=data_format)

        self.gY = tf.placeholder(tf.float32, (None,) + self.output_shape, "out_gradient")

        self.input_sizes = tf.placeholder(tf.int32, [4])
        self.conv_back_input = tf.nn.conv2d_backprop_input(
            self.input_sizes, self.filter, self.gY, [1, stride, stride, 1], "VALID", data_format=data_format)

        self.filter_sizes = tf.constant([filter_height, filter_width, in_channels, out_channels], dtype=tf.int32)
        self.conv_back_filter = tf.nn.conv2d_backprop_filter(
            self.x, self.filter_sizes, self.gY, [1, stride, stride, 1], "VALID", data_format=data_format)

        TF.init()

    def forward(self, value):
        prev_value = self.prev.forward(value)
        if prev_value.shape[1:] != self.in_shape:
            raise "Invalid input format, needed: (batch, channel , height, width)"

        self.value = TF.sess.run(self.conv, feed_dict={
            self.x: prev_value
        })
        return self.value

    def backward(self, gValue, optimizer):
        batch_size = gValue.shape[0]

        prev_gValue = TF.sess.run(self.conv_back_input, feed_dict={
            self.input_sizes: [batch_size, self.in_shape[0], self.in_shape[1], self.in_shape[2]],
            self.gY: gValue
        })

        self.gW = TF.sess.run(self.conv_back_filter, feed_dict={
            self.x: self.prev.value,
            self.gY: gValue
        })
        self.gW = self.gW.transpose(3, 0, 1, 2)

        # optimizer.update(self.w, self.gW * self.learning_rate)
        self.prev.backward(prev_gValue, optimizer)
