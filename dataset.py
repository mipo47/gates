import numpy as np

DATASET_TYPE = np.float32


class DataSet:
    def __init__(self, inputs = None, outputs = None):
        self.inputs = [] if inputs is None else inputs
        self.outputs = [] if outputs is None else outputs

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
        self.inputs = np.array(self.inputs, copy=False, dtype=DATASET_TYPE)
        self.outputs = np.array(self.outputs, copy=False, dtype=DATASET_TYPE)

    def get_batch(self, max_size):
        length = self.length()
        if length <= max_size:
            return self.inputs, self.outputs

        indices = np.random.choice(length, max_size)
        return self.inputs[indices,], self.outputs[indices,]