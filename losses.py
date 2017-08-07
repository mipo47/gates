from gates.gate import *
from gates.simple import *
from gates.activations import Dropout

class Loss(Gate):
    def __init__(self, prev, expected = 0.0):
        super().__init__(prev)
        self.net = None
        self.expected = expected

    def set_expected(self, expected):
        self.expected = expected

    def forward(self, value):
        self.value = self.net.forward(value)
        return self.value

    def backward(self, dValue, optimizer):
        return self.net.backward(dValue, optimizer)

    def validate(self, inputs, outputs):
        # current_expected = self.expected
        self.set_expected(outputs)

        Dropout.Enable = False
        self.forward(inputs)
        Dropout.Enable = True

        # self.set_expected(current_expected) # restore original value
        return self.value # will be set in forward

    def train(self, optimizer, inputs, outputs = None):
        if outputs is not None:
            self.set_expected(outputs)

        loss = self.forward(inputs)
        self.backward(1.0, optimizer)
        return loss


class LossL1(Loss):
    def __init__(self, prev, expected = 0.0):
        super().__init__(prev, expected)
        net = prev
        self.minus = net = Minus(net, expected)  # expect 0 values after this
        self.power = net = Abs(net)
        self.mean  = net = Mean(net)  # gets single number from all previous values
        self.net = net

    def set_expected(self, expected):
        self.minus.constant = -expected


class LossL2(Loss):
    def __init__(self, prev, expected = 0.0):
        super().__init__(prev, expected)
        net = prev
        self.minus = net = Minus(net, expected)  # expect 0 values after this
        self.power = net = Power(net, 2)
        self.mean  = net = Mean(net)  # gets single number from all previous values
        self.net = net

    def set_expected(self, expected):
        self.minus.constant = -expected


# original: https://github.com/martinkersner/cs231n/blob/master/assignment1/softmax.py
class SoftmaxLoss(Loss):
    def __init__(self, prev, expected=None, is_one_hot=True):
        super().__init__(prev, expected)

        self.is_one_hot = is_one_hot
        if expected is not None:
            self.set_expected(expected)

        self.prob_scores = None

    def set_expected(self, expected):
        #TODO: reduce conversion count
        if self.is_one_hot:
            # one hot to index
            # example [[0,1,0],[0,0,1]] to [1,2]
            self.expected = np.argmax(expected, axis=1)
        else:
            self.expected = expected

    def forward(self, value):
        num_train = value.shape[0]

        prev_value = self.prev.forward(value)
        exps = np.exp(prev_value - np.max(prev_value, axis=1, keepdims=True))
        sum = np.sum(exps, axis=1, keepdims=True)
        self.prob_scores = exps / sum

        # normalize
        self.prob_scores = self.prob_scores / np.sum(self.prob_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(self.prob_scores[range(num_train), self.expected])

        loss = np.sum(correct_log_probs)
        loss /= num_train

        self.value = loss
        return self.value

    def backward(self, gValue, optimizer):
        num_train = self.prob_scores.shape[0]
        dscores = self.prob_scores
        dscores[range(num_train), self.expected] -= 1.0

        prev_gValue = gValue * (dscores / num_train)
        self.prev.backward(prev_gValue, optimizer)