import numpy as np


class DataSet:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, input, output):
        self.inputs.append(input)
        self.outputs.append(output)

    def length(self):
        return len(self.inputs)

    def input_count(self):
        return self.inputs.shape[1]

    def output_count(self):
        return self.outputs.shape[1]

    def to_numpy(self):
        self.inputs = np.array(self.inputs, copy=False)
        self.outputs = np.array(self.outputs, copy=False)

    def get_batch(self, max_size):
        length = self.length()
        if length <= max_size:
            return self.inputs, self.outputs

        indices = np.random.choice(length, max_size)
        return self.inputs[indices,], self.outputs[indices,]