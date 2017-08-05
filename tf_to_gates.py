import tensorflow as tf

TYPE = tf.float32

# Global variables
sess = tf.Session()
x = None # default input placeholder


# Activations
Relu = tf.nn.relu
Tanh = tf.nn.tanh
Sigmoid = tf.nn.sigmoid
Softmax = tf.nn.softmax


# Gates
def Gate(size):
    global x
    x = tf.placeholder(TYPE, [None, size])
    return x


def Layer(prev, size, activation=Sigmoid):
    prev_size = int(prev.get_shape()[1])

    w = tf.Variable(tf.random_normal([prev_size, size]), dtype=TYPE)
    b = tf.Variable(tf.random_normal([size]), dtype=TYPE)

    net = tf.add(tf.matmul(prev, w), b)

    if activation is not None:
        net = activation(net)

    return net


# Losses
class SoftmaxLoss:
    def __init__(self, prev):
        self.prev = prev
        self.sess = None
        self.cost = None
        self.x = x
        self.y = None

    def init(self, inputs, outputs):
        if self.cost is not None:
            return

        # n_input = inputs.shape[1]
        n_classes = outputs.shape[1]

        self.y = tf.placeholder(TYPE, [None, n_classes], "outputs")

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prev, labels=self.y))


    def train(self, optimizer, inputs, outputs):
        self.init(inputs, outputs)
        tf_optimizer = optimizer.get_optimizer(self.cost)

        _, c = sess.run([tf_optimizer, self.cost], feed_dict={
            self.x: inputs,
            self.y: outputs
        })
        return c

    def validate(self, inputs, outputs):
        self.init(inputs, outputs)
        return sess.run(self.cost, feed_dict={
            self.x: inputs,
            self.y: outputs
        })


class LossL2(SoftmaxLoss):
    def init(self, inputs, outputs):
        if self.cost is None:
            output_count = outputs.shape[1]
            self.y = tf.placeholder(TYPE, [None, output_count], "outputs")
            self.cost = tf.reduce_mean(tf.pow(self.prev - self.y, 2))


# Optimizers
class Adam:
    def __init__(self):
        self.optimizer = None
        self.cost = None

    def get_optimizer(self, cost):
        if self.cost != cost:
            self.cost = cost
            self.optimizer = tf.train.AdamOptimizer().minimize(cost)
            sess.run(tf.global_variables_initializer()) #TODO: make it somewhere else

        return self.optimizer


# Helpers
class Checkpoint:
    def __init__(self, net):
        self.saver = tf.train.Saver()
        self.path = './temp/tf_checkpoint'

    def backup(self):
        self.saver.save(sess, self.path)

    def restore(self):
        self.saver.restore(sess, self.path)
